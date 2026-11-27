import time
from collections import deque

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import gym
import logging
from arguments import get_args
from env import make_vec_envs
from utils.storage import GlobalRolloutStorage, FIFOMemory
from utils.optimization import get_optimizer
from env.habitat.utils.dynamic_objects import rasterize
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module

import algo

import sys
import matplotlib

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt

# plt.ion()
# fig, ax = plt.subplots(1,4, figsize=(10, 2.5), facecolor="whitesmoke")


args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1.gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]


def main():
    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    policy_loss = 0

    best_cost = 100000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)

    g_masks = torch.ones(num_scenes).float().to(device)
    l_masks = torch.zeros(num_scenes).float().to(device)

    best_local_loss = np.inf
    best_g_reward = -np.inf

    if args.eval:
        traj_lengths = args.max_episode_length // args.num_local_steps
        explored_area_log = np.zeros((num_scenes, num_episodes, traj_lengths))
        explored_ratio_log = np.zeros((num_scenes, num_episodes, traj_lengths))

    # Dynamic-object contact stats per episode:
    # [contact events, contact steps, blocked steps, object yields]
    use_dyn = args.dyn_mode != "none"
    # phantom disks in the global policy input only with -matrixManipulation
    use_dyn_matrix = use_dyn and args.matrixManipulation
    dyn_ep = np.zeros((num_scenes, 4))
    dyn_episode_totals = deque(maxlen=1000)
    if args.eval and use_dyn:
        dyn_log = np.zeros((num_scenes, num_episodes, 4))
        # Perception / map metrics per episode (see dyn_step_metrics):
        # [object cells, detected object cells, steps with an object in view,
        #  steps with >= 50% of its cells detected, ghost cells summed over
        #  steps, ghost cells at the last scored step, number of scored
        #  steps, closest agent-object distance (m)]
        P_OBJ, P_DET, P_VIS, P_VIS_DET, P_GHOST_SUM, P_GHOST_LAST, P_STEPS, \
            P_MIN_DIST = range(8)
        percep_ep = np.zeros((num_scenes, 8))
        percep_ep[:, P_MIN_DIST] = np.inf
        percep_log = np.zeros((num_scenes, num_episodes, 8))

    g_episode_rewards = deque(maxlen=1000)

    l_action_losses = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    # Initialize map variables
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations

    torch.set_grad_enabled(False)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
                       int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, 4, local_w, local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    if args.eval and use_dyn:
        # Cells (full-map frame) that objects have covered this episode, and
        # whether each was already an obstacle in the agent's map before an
        # object first covered it (such cells are not counted as ghosts)
        dyn_trail = np.zeros((num_scenes, full_w, full_h), dtype=bool)
        dyn_pre = np.zeros((num_scenes, full_w, full_h), dtype=bool)

    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def overlay_dyn(infos, est_poses):
        # Max dynamic objects into the global policy input only (a copy of
        # local_map / full_map); the persistent maps are never touched.
        if not use_dyn_matrix:
            return
        for e in range(num_scenes):
            rel = infos[e]['dyn_rel']
            if not rel:
                continue
            dyn_l = rasterize(rel, est_poses[e], (local_w, local_h),
                              (lmb[e, 0], lmb[e, 2]),
                              args.map_resolution, args.dyn_radius)
            dyn_f = rasterize(rel, est_poses[e], (local_w, local_h), (0, 0),
                              args.map_resolution, args.dyn_radius,
                              scale=args.global_downscaling)
            global_input[e, 0] = torch.max(global_input[e, 0],
                                           torch.from_numpy(dyn_l))
            global_input[e, 4] = torch.max(global_input[e, 4],
                                           torch.from_numpy(dyn_f))

    def dyn_step_metrics(infos, proj_pred, prev_obst, obst):
        """Per-step evaluation metrics for dynamic objects.
        detection: of the egocentric cells where a rendered object is seen
          in depth (infos 'dyn_obj_fp', -withRendering only), the share the
          SLAM network's per-frame prediction (proj_pred) marks as obstacle.
        ghosts: obstacle cells in the agent's own map (local_map ch 0) that
          lie on the path objects have covered this episode, but not where
          an object is now, and were not obstacles before an object first
          covered them. Objects are placed from their true offset applied to
          the estimated pose, like the -matrixManipulation overlay."""
        res = args.map_resolution
        for e in range(num_scenes):
            st = infos[e]['dyn_step']
            percep_ep[e, P_MIN_DIST] = min(percep_ep[e, P_MIN_DIST],
                                           st['min_dist'])
            m = infos[e]['dyn_obj_fp']
            if m is not None and m.any():
                det = (proj_pred[e] > 0.5) & m
                percep_ep[e, P_OBJ] += m.sum()
                percep_ep[e, P_DET] += det.sum()
                percep_ep[e, P_VIS] += 1
                percep_ep[e, P_VIS_DET] += det.sum() >= 0.5 * m.sum()

            rel = infos[e]['dyn_rel_all']
            r0, r1, c0, c1 = lmb[e]
            pose = planner_pose_inputs[e, :3]
            core = rasterize(rel, pose, (local_w, local_h), (r0, c0), res,
                             args.dyn_radius) > 0
            # box corners reach dyn_radius * sqrt(2); plus one cell margin
            now = rasterize(rel, pose, (local_w, local_h), (r0, c0), res,
                            args.dyn_radius * 1.42 + res / 100.) > 0
            trail = dyn_trail[e, r0:r1, c0:c1]
            pre = dyn_pre[e, r0:r1, c0:c1]
            new = core & ~trail
            pre[new] = prev_obst[e][new]
            trail |= core
            ghost = int((obst[e] & trail & ~now & ~pre).sum())
            percep_ep[e, P_GHOST_SUM] += ghost
            percep_ep[e, P_GHOST_LAST] = ghost
            percep_ep[e, P_STEPS] += 1

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()

    # Global policy observation space
    g_observation_space = gym.spaces.Box(0, 1,
                                         (8,
                                          local_w,
                                          local_h), dtype='uint8')

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=(2,), dtype=np.float32)

    # Local policy observation space
    l_observation_space = gym.spaces.Box(0, 255,
                                         (3,
                                          args.frame_width,
                                          args.frame_width), dtype='uint8')

    # Local and Global policy recurrent layer sizes
    l_hidden_size = args.local_hidden_size
    g_hidden_size = args.global_hidden_size

    # slam
    nslam_module = Neural_SLAM_Module(args).to(device)
    slam_optimizer = get_optimizer(nslam_module.parameters(),
                                   args.slam_optimizer)

    # Global policy
    g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'downscaling': args.global_downscaling
                                      }).to(device)
    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.global_lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    # Local policy
    l_policy = Local_IL_Policy(l_observation_space.shape, envs.action_space.n,
                               recurrent=args.use_recurrent_local,
                               hidden_size=l_hidden_size,
                               deterministic=args.use_deterministic_local).to(device)
    local_optimizer = get_optimizer(l_policy.parameters(),
                                    args.local_optimizer)

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      1).to(device)

    slam_memory = FIFOMemory(args.slam_memory_size)

    # Loading model
    if args.load_slam != "0":
        print("Loading slam {}".format(args.load_slam))
        state_dict = torch.load(args.load_slam,
                                map_location=lambda storage, loc: storage)
        nslam_module.load_state_dict(state_dict)

    if not args.train_slam:
        nslam_module.eval()

    if args.load_global != "0":
        print("Loading global {}".format(args.load_global))
        state_dict = torch.load(args.load_global,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if not args.train_global:
        g_policy.eval()

    if args.load_local != "0":
        print("Loading local {}".format(args.load_local))
        state_dict = torch.load(args.load_local,
                                map_location=lambda storage, loc: storage)
        l_policy.load_state_dict(state_dict)

    if not args.train_local:
        l_policy.eval()

    # Predict map from frame 1:
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx
         in range(num_scenes)])
    ).float().to(device)

    _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
        nslam_module(obs, obs, poses, local_map[:, 0, :, :],
                     local_map[:, 1, :, :], local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, 8, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:4, :, :] = local_map.detach()
    global_input[:, 4:, :, :] = nn.MaxPool2d(args.global_downscaling)(full_map)
    overlay_dyn(infos, locs + origins)

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(global_orientation)

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states = \
        g_policy.act(
            g_rollouts.obs[0],
            g_rollouts.rec_states[0],
            g_rollouts.masks[0],
            extras=g_rollouts.extras[0],
            deterministic=False
        )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                    for action in cpu_actions]

    # Compute planner inputs
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['goal'] = global_goals[e]
        p_input['map_pred'] = global_input[e, 0, :, :].detach().cpu().numpy()
        p_input['exp_pred'] = global_input[e, 1, :, :].detach().cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]

    # Output stores local goals as well as the the ground-truth action
    output = envs.get_short_term_goal(planner_inputs)

    last_obs = obs.detach()
    local_rec_states = torch.zeros(num_scenes, l_hidden_size).to(device)
    start = time.time()

    total_num_steps = -1
    g_reward = 0

    torch.set_grad_enabled(False)

    for ep_num in range(num_episodes):
        for step in range(args.max_episode_length):
            total_num_steps += 1

            g_step = (step // args.num_local_steps) % args.num_global_steps
            eval_g_step = step // args.num_local_steps + 1
            l_step = step % args.num_local_steps

            # ------------------------------------------------------------------
            # Local Policy
            del last_obs
            last_obs = obs.detach()
            local_masks = l_masks
            local_goals = output[:, :-1].to(device).long()

            if args.train_local:
                torch.set_grad_enabled(True)

            action, action_prob, local_rec_states = l_policy(
                obs,
                local_rec_states,
                local_masks,
                extras=local_goals,
            )

            if args.train_local:
                action_target = output[:, -1].long().to(device)
                policy_loss += nn.CrossEntropyLoss()(action_prob, action_target)
                torch.set_grad_enabled(False)
            l_action = action.cpu()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Env step
            obs, rew, done, infos = envs.step(l_action)

            l_masks = torch.FloatTensor([0 if x else 1
                                         for x in done]).to(device)
            g_masks *= l_masks

            if use_dyn:
                for e in range(num_scenes):
                    s = infos[e]['dyn_step']
                    dyn_ep[e] += [s['event'], s['contact'], s['blocked'],
                                  s['yields']]
                if step == args.max_episode_length - 1:
                    if args.eval:
                        dyn_log[:, ep_num] = dyn_ep
                        # last step's infos come from the auto-reset (next
                        # episode): keep only its carried-over min_dist
                        for e in range(num_scenes):
                            percep_ep[e, P_MIN_DIST] = min(
                                percep_ep[e, P_MIN_DIST],
                                infos[e]['dyn_step']['min_dist'])
                        percep_log[:, ep_num] = percep_ep
                        percep_ep.fill(0.)
                        percep_ep[:, P_MIN_DIST] = np.inf
                        dyn_trail.fill(False)
                        dyn_pre.fill(False)
                    for e in range(num_scenes):
                        dyn_episode_totals.append(dyn_ep[e].copy())
                    dyn_ep.fill(0.)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Reinitialize variables when episode ends
            if step == args.max_episode_length - 1:  # Last episode step
                init_map_and_pose()
                del last_obs
                last_obs = obs.detach()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Neural SLAM Module
            if args.train_slam:
                # Add frames to memory
                for env_idx in range(num_scenes):
                    env_obs = obs[env_idx].to("cpu")
                    env_poses = torch.from_numpy(np.asarray(
                        infos[env_idx]['sensor_pose']
                    )).float().to("cpu")
                    env_gt_fp_projs = torch.from_numpy(np.asarray(
                        infos[env_idx]['fp_proj']
                    )).unsqueeze(0).float().to("cpu")
                    env_gt_fp_explored = torch.from_numpy(np.asarray(
                        infos[env_idx]['fp_explored']
                    )).unsqueeze(0).float().to("cpu")
                    env_gt_pose_err = torch.from_numpy(np.asarray(
                        infos[env_idx]['pose_err']
                    )).float().to("cpu")
                    slam_memory.push(
                        (last_obs[env_idx].cpu(), env_obs, env_poses),
                        (env_gt_fp_projs, env_gt_fp_explored, env_gt_pose_err))

            poses = torch.from_numpy(np.asarray(
                [infos[env_idx]['sensor_pose'] for env_idx
                 in range(num_scenes)])
            ).float().to(device)

            score_dyn = args.eval and use_dyn and \
                step != args.max_episode_length - 1
            if score_dyn:
                prev_obst = (local_map[:, 0] > 0.5).cpu().numpy()
            proj_pred, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, \
                local_pose = nslam_module(last_obs, obs, poses,
                                          local_map[:, 0, :, :],
                                          local_map[:, 1, :, :], local_pose,
                                          build_maps=True)

            locs = local_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs + origins
            if score_dyn:
                dyn_step_metrics(infos, proj_pred[:, 0].cpu().numpy(),
                                 prev_obst,
                                 (local_map[:, 0] > 0.5).cpu().numpy())
            local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                local_map[e, 2:, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Global Policy
            if l_step == args.num_local_steps - 1:
                # For every global step, update the full and local maps
                for e in range(num_scenes):
                    full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                        local_map[e]
                    full_pose[e] = local_pose[e] + \
                                   torch.from_numpy(origins[e]).to(device).float()

                    locs = full_pose[e].cpu().numpy()
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]

                    lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                      (local_w, local_h),
                                                      (full_w, full_h))

                    planner_pose_inputs[e, 3:] = lmb[e]
                    origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                                  lmb[e][0] * args.map_resolution / 100.0, 0.]

                    local_map[e] = full_map[e, :,
                                   lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
                    local_pose[e] = full_pose[e] - \
                                    torch.from_numpy(origins[e]).to(device).float()

                locs = local_pose.cpu().numpy()
                for e in range(num_scenes):
                    global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
                global_input[:, 0:4, :, :] = local_map
                global_input[:, 4:, :, :] = \
                    nn.MaxPool2d(args.global_downscaling)(full_map)
                overlay_dyn(infos, full_pose.cpu().numpy())

                if False:
                    for i in range(4):
                        ax[i].clear()
                        ax[i].set_yticks([])
                        ax[i].set_xticks([])
                        ax[i].set_yticklabels([])
                        ax[i].set_xticklabels([])
                        ax[i].imshow(global_input.cpu().numpy()[0, 4 + i])
                    plt.gcf().canvas.flush_events()
                    # plt.pause(0.1)
                    fig.canvas.start_event_loop(0.001)
                    plt.gcf().canvas.flush_events()

                # Get exploration reward and metrics
                g_reward = torch.from_numpy(np.asarray(
                    [infos[env_idx]['exp_reward'] for env_idx
                     in range(num_scenes)])
                ).float().to(device)

                if args.eval:
                    g_reward = g_reward*50.0 # Convert reward to area in m2

                g_process_rewards += g_reward.cpu().numpy()
                g_total_rewards = g_process_rewards * \
                                  (1 - g_masks.cpu().numpy())
                g_process_rewards *= g_masks.cpu().numpy()
                per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

                if np.sum(g_total_rewards) != 0:
                    for tr in g_total_rewards:
                        g_episode_rewards.append(tr) if tr != 0 else None

                if args.eval:
                    exp_ratio = torch.from_numpy(np.asarray(
                        [infos[env_idx]['exp_ratio'] for env_idx
                         in range(num_scenes)])
                    ).float()

                    for e in range(num_scenes):
                        explored_area_log[e, ep_num, eval_g_step - 1] = \
                            explored_area_log[e, ep_num, eval_g_step - 2] + \
                            g_reward[e].cpu().numpy()
                        explored_ratio_log[e, ep_num, eval_g_step - 1] = \
                            explored_ratio_log[e, ep_num, eval_g_step - 2] + \
                            exp_ratio[e].cpu().numpy()

                # Add samples to global policy storage
                g_rollouts.insert(
                    global_input, g_rec_states,
                    g_action, g_action_log_prob, g_value,
                    g_reward, g_masks, global_orientation
                )

                # Sample long-term goal from global policy
                g_value, g_action, g_action_log_prob, g_rec_states = \
                    g_policy.act(
                        g_rollouts.obs[g_step + 1],
                        g_rollouts.rec_states[g_step + 1],
                        g_rollouts.masks[g_step + 1],
                        extras=g_rollouts.extras[g_step + 1],
                        deterministic=False
                    )
                cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
                global_goals = [[int(action[0] * local_w),
                                 int(action[1] * local_h)]
                                for action in cpu_actions]

                g_reward = 0
                g_masks = torch.ones(num_scenes).float().to(device)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Get short term goal
            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = planner_pose_inputs[e]
                p_input['goal'] = global_goals[e]

            output = envs.get_short_term_goal(planner_inputs)
            # ------------------------------------------------------------------

            ### TRAINING
            torch.set_grad_enabled(True)
            # ------------------------------------------------------------------
            # Train Neural SLAM Module
            if args.train_slam and len(slam_memory) > args.slam_batch_size:
                for _ in range(args.slam_iterations):
                    inputs, outputs = slam_memory.sample(args.slam_batch_size)
                    b_obs_last, b_obs, b_poses = inputs
                    gt_fp_projs, gt_fp_explored, gt_pose_err = outputs

                    b_obs = b_obs.to(device)
                    b_obs_last = b_obs_last.to(device)
                    b_poses = b_poses.to(device)

                    gt_fp_projs = gt_fp_projs.to(device)
                    gt_fp_explored = gt_fp_explored.to(device)
                    gt_pose_err = gt_pose_err.to(device)

                    b_proj_pred, b_fp_exp_pred, _, _, b_pose_err_pred, _ = \
                        nslam_module(b_obs_last, b_obs, b_poses,
                                     None, None, None,
                                     build_maps=False)
                    loss = 0
                    if args.proj_loss_coeff > 0:
                        proj_loss = F.binary_cross_entropy(b_proj_pred,
                                                           gt_fp_projs)
                        costs.append(proj_loss.item())
                        loss += args.proj_loss_coeff * proj_loss

                    if args.exp_loss_coeff > 0:
                        exp_loss = F.binary_cross_entropy(b_fp_exp_pred,
                                                          gt_fp_explored)
                        exp_costs.append(exp_loss.item())
                        loss += args.exp_loss_coeff * exp_loss

                    if args.pose_loss_coeff > 0:
                        pose_loss = torch.nn.MSELoss()(b_pose_err_pred,
                                                       gt_pose_err)
                        pose_costs.append(args.pose_loss_coeff *
                                          pose_loss.item())
                        loss += args.pose_loss_coeff * pose_loss

                    if args.train_slam:
                        slam_optimizer.zero_grad()
                        loss.backward()
                        slam_optimizer.step()

                    del b_obs_last, b_obs, b_poses
                    del gt_fp_projs, gt_fp_explored, gt_pose_err
                    del b_proj_pred, b_fp_exp_pred, b_pose_err_pred

            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Train Local Policy
            if (l_step + 1) % args.local_policy_update_freq == 0 \
                    and args.train_local:
                local_optimizer.zero_grad()
                policy_loss.backward()
                local_optimizer.step()
                l_action_losses.append(policy_loss.item())
                policy_loss = 0
                local_rec_states = local_rec_states.detach_()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Train Global Policy
            if g_step % args.num_global_steps == args.num_global_steps - 1 \
                    and l_step == args.num_local_steps - 1:
                if args.train_global:
                    g_next_value = g_policy.get_value(
                        g_rollouts.obs[-1],
                        g_rollouts.rec_states[-1],
                        g_rollouts.masks[-1],
                        extras=g_rollouts.extras[-1]
                    ).detach()

                    g_rollouts.compute_returns(g_next_value, args.use_gae,
                                               args.gamma, args.tau)
                    g_value_loss, g_action_loss, g_dist_entropy = \
                        g_agent.update(g_rollouts)
                    g_value_losses.append(g_value_loss)
                    g_action_losses.append(g_action_loss)
                    g_dist_entropies.append(g_dist_entropy)
                g_rollouts.after_update()
            # ------------------------------------------------------------------

            # Finish Training
            torch.set_grad_enabled(False)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Logging
            if total_num_steps % args.log_interval == 0:
                end = time.time()
                time_elapsed = time.gmtime(end - start)
                log = " ".join([
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(total_num_steps *
                                               num_scenes),
                    "FPS {},".format(int(total_num_steps * num_scenes \
                                         / (end - start)))
                ])

                log += "\n\tRewards:"

                if len(g_episode_rewards) > 0:
                    log += " ".join([
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards),
                            np.median(per_step_g_rewards)),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards))
                    ])

                if use_dyn and len(dyn_episode_totals) > 0:
                    d = np.mean(dyn_episode_totals, axis=0)
                    log += " ".join([
                        "\n\tDyn objects per eps events/contact/blocked/yield:",
                        "{:.2f}/{:.2f}/{:.2f}/{:.2f},".format(*d)
                    ])

                log += "\n\tLosses:"

                if args.train_local and len(l_action_losses) > 0:
                    log += " ".join([
                        " Local Loss:",
                        "{:.3f},".format(
                            np.mean(l_action_losses))
                    ])

                if args.train_global and len(g_value_losses) > 0:
                    log += " ".join([
                        " Global Loss value/action/dist:",
                        "{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies))
                    ])

                if args.train_slam and len(costs) > 0:
                    log += " ".join([
                        " SLAM Loss proj/exp/pose:"
                        "{:.4f}/{:.4f}/{:.4f}".format(
                            np.mean(costs),
                            np.mean(exp_costs),
                            np.mean(pose_costs))
                    ])

                print(log)
                logging.info(log)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Save best models
            if (total_num_steps * num_scenes) % args.save_interval < \
                    num_scenes:

                # Save Neural SLAM Model
                if len(costs) >= 1000 and np.mean(costs) < best_cost \
                        and not args.eval:
                    best_cost = np.mean(costs)
                    torch.save(nslam_module.state_dict(),
                               os.path.join(log_dir, "model_best.slam"))

                # Save Local Policy Model
                if len(l_action_losses) >= 100 and \
                        (np.mean(l_action_losses) <= best_local_loss) \
                        and not args.eval:
                    torch.save(l_policy.state_dict(),
                               os.path.join(log_dir, "model_best.local"))

                    best_local_loss = np.mean(l_action_losses)

                # Save Global Policy Model
                if len(g_episode_rewards) >= 100 and \
                        (np.mean(g_episode_rewards) >= best_g_reward) \
                        and not args.eval:
                    torch.save(g_policy.state_dict(),
                               os.path.join(log_dir, "model_best.global"))
                    best_g_reward = np.mean(g_episode_rewards)

            # Save periodic models
            if (total_num_steps * num_scenes) % args.save_periodic < \
                    num_scenes:
                step = total_num_steps * num_scenes
                if args.train_slam:
                    torch.save(nslam_module.state_dict(),
                               os.path.join(dump_dir,
                                            "periodic_{}.slam".format(step)))
                if args.train_local:
                    torch.save(l_policy.state_dict(),
                               os.path.join(dump_dir,
                                            "periodic_{}.local".format(step)))
                if args.train_global:
                    torch.save(g_policy.state_dict(),
                               os.path.join(dump_dir,
                                            "periodic_{}.global".format(step)))
            # ------------------------------------------------------------------

    # Print and save model performance numbers during evaluation
    if args.eval:
        logfile = open("{}/explored_area.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_area_log[e].shape[0]):
                logfile.write(str(explored_area_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        logfile = open("{}/explored_ratio.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_ratio_log[e].shape[0]):
                logfile.write(str(explored_ratio_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        log = "Final Exp Area: \n"
        for i in range(explored_area_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_area_log[:, :, i]))

        log += "\nFinal Exp Ratio: \n"
        for i in range(explored_ratio_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_ratio_log[:, :, i]))

        if use_dyn:
            with open("{}/dyn_collisions.txt".format(dump_dir), "w+") as f:
                f.write("# process episode events contact_steps "
                        "blocked_steps object_yields\n")
                for e in range(num_scenes):
                    for i in range(num_episodes):
                        f.write("{} {} {:d} {:d} {:d} {:d}\n".format(
                            e, i, *dyn_log[e, i].astype(int)))
            flat = dyn_log.reshape(-1, 4)
            log += "\nDyn objects ({}, n={}, block={}, fov_only={}):".format(
                args.dyn_mode, args.num_dyn_obj, args.dyn_block,
                args.dyn_fov_only)
            log += "\n  contact events/eps: {:.3f}".format(flat[:, 0].mean())
            log += "\n  contact steps/eps:  {:.3f}".format(flat[:, 1].mean())
            log += "\n  blocked steps/eps:  {:.3f}".format(flat[:, 2].mean())
            log += "\n  object yields/eps:  {:.3f}".format(flat[:, 3].mean())
            log += "\n  collision-free eps: {:.1f}%".format(
                100. * (flat[:, 0] == 0).mean())

            cell_m2 = (args.map_resolution / 100.) ** 2
            pl = percep_log.reshape(-1, 8)
            steps = np.maximum(pl[:, P_STEPS], 1)
            md = pl[:, P_MIN_DIST]
            with open("{}/dyn_perception.txt".format(dump_dir), "w+") as f:
                f.write("# process episode object_cells detected_cells "
                        "steps_in_view steps_detected ghost_mean_m2 "
                        "ghost_last_m2 min_dist_m\n")
                for e in range(num_scenes):
                    for i in range(num_episodes):
                        r = percep_log[e, i]
                        f.write("{} {} {:d} {:d} {:d} {:d} {:.4f} {:.4f} "
                                "{:.3f}\n".format(
                                    e, i, *r[:4].astype(int),
                                    r[P_GHOST_SUM] / max(r[P_STEPS], 1)
                                    * cell_m2,
                                    r[P_GHOST_LAST] * cell_m2,
                                    r[P_MIN_DIST]))
            log += "\nDyn perception (matrix={}, rendering={}):".format(
                args.matrixManipulation, args.withRendering)
            if pl[:, P_OBJ].sum() > 0:
                log += "\n  object cell recall:   {:.3f}".format(
                    pl[:, P_DET].sum() / pl[:, P_OBJ].sum())
                log += "\n  frame detection rate: {:.3f}".format(
                    pl[:, P_VIS_DET].sum() / max(pl[:, P_VIS].sum(), 1))
                log += "\n  steps in view/eps:    {:.1f}".format(
                    pl[:, P_VIS].mean())
            else:
                log += "\n  detection: n/a (objects not rendered)"
            log += "\n  ghost area/eps (mean over steps): {:.3f} m2".format(
                (pl[:, P_GHOST_SUM] / steps).mean() * cell_m2)
            log += "\n  ghost area/eps (last step):       {:.3f} m2".format(
                pl[:, P_GHOST_LAST].mean() * cell_m2)
            finite = np.isfinite(md)
            if finite.any():
                log += "\n  closest approach/eps: {:.3f} m".format(
                    md[finite].mean())

        print(log)
        logging.info(log)


if __name__ == "__main__":
    main()
