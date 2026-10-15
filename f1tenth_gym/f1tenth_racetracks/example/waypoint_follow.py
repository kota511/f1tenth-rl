# waypoint_follow.py
import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace
from track_utils import compute_track_length
import json

from numba import njit

from pyglet.gl import GL_POINTS

"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.
    
        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)
    
    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.
    
    Assumes that the first segment passes within a single radius of the point
    
    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb, max_speed=None):
        self.wheelbase = wb
        self.conf = conf
        self.max_speed = max_speed  # Added maximum speed parameter
        self.load_waypoints(conf)
        self.max_reacquire = 20.
        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
    
    def get_track_length(self, closed_loop=True):
        """
        Computes the total track length based on the loaded waypoints.
        If closed_loop is True, adds the final segment connecting the last waypoint to the first.
        """
        pts = np.vstack((self.waypoints[:, self.conf.wpt_xind],
                         self.waypoints[:, self.conf.wpt_yind])).T
        diffs = np.diff(pts, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(segment_lengths)
        
        if closed_loop:
            closing_dist = np.linalg.norm(pts[0] - pts[-1])
            total_length += closing_dist
        
        return total_length

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        scaled_points = 50.*points
        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None,
                                ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
        
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind],
                          self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 is None:
                return None
            current_waypoint = np.empty((3, ))
            current_waypoint[0:2] = wpts[i2, :]
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed
        # Apply the maximum speed cap
        speed = min(speed, self.max_speed)
        return speed, steering_angle


class FlippyPlanner:
    """
    Planner designed to exploit integration methods and dynamics.
    For testing only. To observe this error, use single track dynamics for all velocities >0.1
    """
    def __init__(self, speed=1, flip_every=1, steer=2):
        self.speed = speed
        self.flip_every = flip_every
        self.counter = 0
        self.steer = steer
    
    def render_waypoints(self, *args, **kwargs):
        pass

    def plan(self, *args, **kwargs):
        if self.counter % self.flip_every == 0:
            self.counter = 0
            self.steer *= -1
        return self.speed, self.steer


def main():
    with open("center_config_example_map.yaml") as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)

    conf2 = Namespace(**conf_dict)
    conf2.wpt_path = conf_dict['wpt_path2']
    conf2.sx = conf_dict['sx2']
    conf2.sy = conf_dict['sy2']
    conf2.stheta = conf_dict['stheta2']

    wheelbase = 0.17145 + 0.15875
    max_speed1 = 6.0
    max_speed2 = 7.0
    work = {'tlad': conf_dict['tlad'], 'vgain': conf_dict['vgain']}

    planner1 = PurePursuitPlanner(conf, wheelbase, max_speed1)
    planner2 = PurePursuitPlanner(conf2, wheelbase, max_speed2)

    def render_callback(env_renderer):
        planner1.render_waypoints(env_renderer)
        planner2.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0',
                   map=conf.map_path,
                   map_ext=conf.map_ext,
                   num_agents=2,
                   timestep=0.01,
                   integrator=Integrator.RK4)
    env.add_render_callback(render_callback)

    poses = np.array([
        [conf.sx, conf.sy, conf.stheta],
        [conf2.sx, conf2.sy, conf2.stheta]
    ])
    obs, _, done, _ = env.reset(poses=poses)
    env.render()

    timestep = env.timestep
    sim_time = 0.0

    # Define finish line using initial x positions
    start_x = [conf.sx, conf2.sx]
    start_y = [conf.sy, conf2.sy]
    finish_threshold = 0.3  # meters

    finished = [False, False]
    laptimes = [0.0, 0.0]

    while not all(finished):
        sim_time += timestep

        speed1, steer1 = planner1.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        speed2, steer2 = planner2.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], work['tlad'], work['vgain'])

        actions = np.array([[steer1, speed1], [steer2, speed2]])
        obs, _, done, _ = env.step(actions)

        for i in range(2):
            if not finished[i]:
                dx = obs['poses_x'][i] - start_x[i]
                dy = obs['poses_y'][i] - start_y[i]
                dist = np.sqrt(dx**2 + dy**2)
                # Check if agent returned near start after traveling some distance
                if dist < finish_threshold and sim_time > 2.0:  # wait at least 2s to avoid false positives
                    finished[i] = True
                    laptimes[i] = sim_time

        env.render()

    print(f"Agent 1 lap time: {round(laptimes[0], 3)} seconds")
    print(f"Agent 2 lap time: {round(laptimes[1], 3)} seconds")


if __name__ == '__main__':
    main()