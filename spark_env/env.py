import numpy as np
import copy
from collections import OrderedDict
from utils import *
from spark_env.action_map import compute_act_map, get_frontier_acts
from spark_env.reward_calculator import RewardCalculator
from spark_env.moving_executors import MovingExecutors
from spark_env.executor_commit import ExecutorCommit
from spark_env.free_executors import FreeExecutors
from spark_env.wall_time import WallTime
from spark_env.timeline import Timeline
from spark_env.executor import Executor
from spark_env.job_dag import JobDAG
from spark_env.task import Task
from spark_env.node import Node


class Environment(object):
    def __init__(self, exec_cap=None, carbon_schedule=None, job_dags=None):

        # isolated random number generator
        self.np_random = np.random.RandomState()

        # global timer
        self.wall_time = WallTime()

        # uses priority queue
        self.timeline = Timeline()

        # executors
        self.executors = OrderedSet()
        for exec_id in range(exec_cap):
            self.executors.add(Executor(exec_id))

        # free executors
        self.free_executors = FreeExecutors(self.executors)

        # moving executors
        self.moving_executors = MovingExecutors()

        # moving delay
        self.moving_delay = 0

        # executor commit
        self.exec_commit = ExecutorCommit()

        # prevent agent keeps selecting the same node
        self.node_selected = set()

        # for computing reward at each step
        self.reward_calculator = RewardCalculator()

        if carbon_schedule is None:
            self.carbon_schedule = {0: 0.5, np.inf: 0.5}
        else:
            self.carbon_schedule = carbon_schedule
        
        self.current_carbon_intensity = self.get_carbon_intensity(self.wall_time.curr_time)

        # Initialize job DAGs if provided
        self.job_dags = OrderedSet(job_dags) if job_dags else OrderedSet()
        self.action_map = compute_act_map(self.job_dags)
        for job_dag in self.job_dags:
            self.add_job(job_dag)
        self.source_job = None
        self.num_source_exec = len(self.executors)
        self.exec_to_schedule = OrderedSet(self.executors)

    def get_carbon_intensity(self, current_time):
        # round the current time to the nearest (previous) time in the keys
        keys = [key for key in self.carbon_schedule.keys() if key < current_time]
        if len(keys) > 0:
            current_CI_time = max(keys)
            return self.carbon_schedule[current_CI_time]
        # else just return the first value
        return self.carbon_schedule[list(self.carbon_schedule.keys())[0]]


    # def compute_carbon_usage(self, start_time, end_time):
    #     carbon_usage = 0.0
    #     CI_time = self.last_CI_time  # Start from the last used interval
    #     keys = sorted(self.carbon_schedule.keys())  # Sorted list of keys for fixed intervals
    #     num_intervals = len(keys)
        
    #     # Update pointer to the start interval for current `start_time`
    #     while CI_time < num_intervals - 1 and keys[CI_time] < start_time:
    #         CI_time += 1
    #     self.last_CI_time = CI_time  # Update the pointer

    #     # Loop through intervals until end time
    #     while CI_time < num_intervals - 1 and keys[CI_time] < end_time:
    #         # Current intensity and interval bounds
    #         intensity = self.carbon_schedule[keys[CI_time]]
    #         next_CI_time = keys[CI_time + 1]  # Next interval start

    #         # Calculate overlap
    #         overlap_start = max(start_time, keys[CI_time])
    #         overlap_end = min(end_time, next_CI_time)
    #         duration = overlap_end - overlap_start
    #         carbon_usage += duration * intensity

    #         # Move to the next interval
    #         CI_time += 1
    #     return carbon_usage
    
    def compute_carbon_usage(self, start_time, end_time):
        return 0.0 # making this a stub to speed things up


    def add_job(self, job_dag):
        self.moving_executors.add_job(job_dag)
        self.free_executors.add_job(job_dag)
        self.exec_commit.add_job(job_dag)

    def assign_executor(self, executor, frontier_changed):
        if executor.node is not None and not executor.node.no_more_tasks:
            # keep working on the previous node
            task = executor.node.schedule(executor)
            self.timeline.push(task.finish_time, task)
        else:
            # need to move on to other nodes
            if frontier_changed:
                # frontier changed, need to consult all free executors
                # note: executor.job_dag might change after self.schedule()
                source_job = executor.job_dag
                if len(self.exec_commit[executor.node]) > 0:
                    # directly fulfill the commitment
                    self.exec_to_schedule = {executor}
                    self.schedule()
                else:
                    # free up the executor
                    self.free_executors.add(source_job, executor)
                # then consult all free executors
                self.exec_to_schedule = OrderedSet(self.free_executors[source_job])
                self.source_job = source_job
                self.num_source_exec = len(self.free_executors[source_job])
            else:
                # just need to schedule one current executor
                self.exec_to_schedule = {executor}
                # only care about executors on the node
                if len(self.exec_commit[executor.node]) > 0:
                    # directly fulfill the commitment
                    self.schedule()
                else:
                    # need to consult for ALL executors on the node
                    # Note: self.exec_to_schedule is immediate
                    #       self.num_source_exec is for commit
                    #       so len(self.exec_to_schedule) !=
                    #       self.num_source_exec can happen
                    self.source_job = executor.job_dag
                    self.num_source_exec = len(executor.node.executors)

    def backup_schedule(self, executor):
        # This function is triggered very rarely. A random policy
        # or the learned polici in early iterations might decide
        # to schedule no executors to any job. This function makes
        # sure the cluster is work conservative. Since the backup
        # policy is not strong, the learning agent should learn to
        # not rely on it.
        backup_scheduled = False
        if executor.job_dag is not None:
            # first try to schedule on current job
            for node in executor.job_dag.frontier_nodes:
                if not self.saturated(node):
                    # greedily schedule a frontier node
                    task = node.schedule(executor)
                    self.timeline.push(task.finish_time, task)
                    backup_scheduled = True
                    break
        # then try to schedule on any available node
        if not backup_scheduled:
            schedulable_nodes = self.get_frontier_nodes()
            if len(schedulable_nodes) > 0:
                node = next(iter(schedulable_nodes))
                self.timeline.push(
                    self.wall_time.curr_time + self.moving_delay, executor)
                # keep track of moving executors
                self.moving_executors.add(executor, node)
                backup_scheduled = True
        # at this point if nothing available, leave executor idle
        if not backup_scheduled:
            self.free_executors.add(executor.job_dag, executor)

    def get_frontier_nodes(self):
        # frontier nodes := unsaturated nodes with all parent nodes saturated
        frontier_nodes = OrderedSet()
        for job_dag in self.job_dags:
            for node in job_dag.nodes:
                if not node in self.node_selected and not self.saturated(node):
                    parents_saturated = True
                    for parent_node in node.parent_nodes:
                        if not self.saturated(parent_node):
                            parents_saturated = False
                            break
                    if parents_saturated:
                        frontier_nodes.add(node)

        return frontier_nodes

    def get_executor_limits(self):
        # "minimum executor limit" for each job
        # executor limit := {job_dag -> int}
        executor_limit = {}

        for job_dag in self.job_dags:

            if self.source_job == job_dag:
                curr_exec = self.num_source_exec
            else:
                curr_exec = 0

            # note: this does not count in the commit and moving executors
            executor_limit[job_dag] = len(job_dag.executors) - curr_exec

        return executor_limit

    def observe(self):
        return self.job_dags, self.source_job, self.num_source_exec, \
               self.get_frontier_nodes(), self.get_executor_limits(), \
               self.exec_commit, self.moving_executors, self.action_map, self.wall_time.curr_time

    def saturated(self, node):
        # frontier nodes := unsaturated nodes with all parent nodes saturated
        anticipated_task_idx = node.next_task_idx + \
           self.exec_commit.node_commit[node] + \
           self.moving_executors.count(node)
        # note: anticipated_task_idx can be larger than node.num_tasks
        # when the tasks finish very fast before commitments are fulfilled
        return anticipated_task_idx >= node.num_tasks

    def schedule(self):
        
        executor = next(iter(self.exec_to_schedule))
        source = executor.job_dag if executor.node is None else executor.node

        # schedule executors from the source until the commitment is fulfilled
        while len(self.exec_commit[source]) > 0 and \
              len(self.exec_to_schedule) > 0:

            # keep fulfilling the commitment using free executors
            node = self.exec_commit.pop(source)
            executor = self.exec_to_schedule.pop()

            # mark executor as in use if it was free executor previously
            if self.free_executors.contain_executor(executor.job_dag, executor):
                self.free_executors.remove(executor)

            if node is None:
                # the next node is explicitly silent, make executor ilde
                if executor.job_dag is not None and \
                   any([not n.no_more_tasks for n in \
                        executor.job_dag.nodes]):
                    # mark executor as idle in its original job
                    self.free_executors.add(executor.job_dag, executor)
                else:
                    # no where to assign, put executor in null pool
                    self.free_executors.add(None, executor)


            elif not node.no_more_tasks:
                # node is not currently saturated
                if executor.job_dag == node.job_dag:
                    # executor local to the job
                    if node in node.job_dag.frontier_nodes:
                        # node is immediately runnable
                        task = node.schedule(executor)
                        self.timeline.push(task.finish_time, task)
                    else:
                        # put executor back in the free pool
                        self.free_executors.add(executor.job_dag, executor)

                else:
                    # need to move executor
                    self.timeline.push(
                        self.wall_time.curr_time + self.moving_delay, executor)
                    # keep track of moving executors
                    self.moving_executors.add(executor, node)

            else:
                # node is already saturated, use backup logic
                self.backup_schedule(executor)

    def step(self, next_node, limit, carbon_aware=False):
        # Initialize variables to track carbon usage intervals
        last_event_time = self.wall_time.curr_time
        last_task_was_dummy = False

        if next_node is None and carbon_aware:
            # Schedule a dummy node to wait for a lower carbon intensity period
            duration = min(self.time_until_next_carbon_reading(), self.time_until_next_scheduling_decision())
            if duration == float('inf'):
                duration = 0

            next_node = self.create_dummy_node(duration, limit)
            dummy_job_dag = next_node.job_dag
            self.add_job(dummy_job_dag)

        # Mark the node as selected
        assert next_node not in self.node_selected
        self.node_selected.add(next_node)
        # Commit the source executor
        executor = next(iter(self.exec_to_schedule))
        source = executor.job_dag if executor.node is None else executor.node

        # Compute number of valid executors to assign
        if next_node is not None:
            use_exec = min(
                next_node.num_tasks - next_node.next_task_idx -
                self.exec_commit.node_commit[next_node] -
                self.moving_executors.count(next_node), limit)
        else:
            use_exec = limit
        assert use_exec > 0

        self.exec_commit.add(source, next_node, use_exec)
        # Deduct the executors that know the destination
        self.num_source_exec -= use_exec
        assert self.num_source_exec >= 0

        if self.num_source_exec == 0:
            # Now a new scheduling round, clean up node selection
            self.node_selected.clear()
            # All commitments are made, now schedule free executors
            self.schedule()

        # Now run to the next event in the virtual timeline
        while len(self.timeline) > 0 and self.num_source_exec == 0:
            # Consult agent by putting executors in source_exec

            new_time, obj = self.timeline.pop()
            # Calculate the time difference since the last event
            time_diff = new_time - last_event_time
            # Update carbon usage only if the last task was not a dummy
            if time_diff > 0:
                carbon_usage = self.compute_carbon_usage(last_event_time, new_time)
                if not last_task_was_dummy:
                    # Accumulate carbon usage for real tasks
                    self.total_carbon_usage += carbon_usage
                else:
                    # Subtract carbon usage associated with dummy tasks
                    self.total_carbon_usage -= carbon_usage

            # Update the wall time
            self.wall_time.update_time(new_time)
            last_event_time = new_time  # Update the last event time

            # Process the event
            if isinstance(obj, Task):  # Task completion event
                finished_task = obj
                node = finished_task.node
                node.num_finished_tasks += 1

                # Check if the task was a dummy task
                last_task_was_dummy = (node.job_dag.name == "dummy")

                # Bookkeepings for node completion
                frontier_changed = False
                if node.num_finished_tasks == node.num_tasks:
                    assert not node.tasks_all_done  # Only complete once
                    node.tasks_all_done = True
                    if node.job_dag is not None:
                        node.job_dag.num_nodes_done += 1
                        frontier_changed = node.job_dag.update_frontier_nodes(node)
                    node.node_finish_time = self.wall_time.curr_time

                # Assign new destination for the executor
                self.assign_executor(finished_task.executor, frontier_changed)

                # Bookkeepings for job completion
                if node.job_dag.num_nodes_done == node.job_dag.num_nodes:
                    assert not node.job_dag.completed  # Only complete once
                    node.job_dag.completed = True
                    node.job_dag.completion_time = self.wall_time.curr_time
                    self.remove_job(node.job_dag)

            elif isinstance(obj, JobDAG):  # New job arrival event
                job_dag = obj
                assert not job_dag.arrived
                job_dag.arrived = True
                # Inform agent about job arrival when stream is enabled
                self.job_dags.add(job_dag)
                self.add_job(job_dag)
                self.action_map = compute_act_map(self.job_dags)
                # Assign free executors (if any) to the new job
                if len(self.free_executors[None]) > 0:
                    self.exec_to_schedule = OrderedSet(self.free_executors[None])
                    self.source_job = None
                    self.num_source_exec = len(self.free_executors[None])

                last_task_was_dummy = False  # Job arrival is not a dummy task

            elif isinstance(obj, Executor):  # Executor arrival event
                executor = obj
                # Pop destination from the tracking record
                node = self.moving_executors.pop(executor)

                if node is not None:
                    # The job is not yet done when executor arrives
                    executor.job_dag = node.job_dag
                    node.job_dag.executors.add(executor)

                if node is not None and not node.no_more_tasks:
                    # The node is still schedulable
                    if node in node.job_dag.frontier_nodes:
                        # Node is immediately runnable
                        task = node.schedule(executor)
                        self.timeline.push(task.finish_time, task)
                    else:
                        # Free up the executor in this job
                        self.free_executors.add(executor.job_dag, executor)
                else:
                    # The node is saturated or the job is done
                    # by the time the executor arrives, use backup logic
                    self.backup_schedule(executor)

                # Check if the executor was moving towards a dummy node
                last_task_was_dummy = (node.job_dag.name == "dummy") if node else False

            else:
                print("illegal event type")
                exit(1)

        # Compute reward
        reward = self.reward_calculator.get_reward(
            self.job_dags, self.wall_time.curr_time)
        if carbon_aware:
            reward = 0

        # No more decision to make, jobs all done or time is up
        done = (self.num_source_exec == 0) and (
            (len(self.timeline) == 0) or
            (self.wall_time.curr_time >= self.max_time)
        )

        if done:
            assert self.wall_time.curr_time >= self.max_time or \
                len(self.job_dags) == 0

        return self.observe(), reward, done

    def remove_job(self, job_dag):
        for executor in list(job_dag.executors):
            executor.detach_job()
        self.exec_commit.remove_job(job_dag)
        self.free_executors.remove_job(job_dag)
        self.moving_executors.remove_job(job_dag)
        if job_dag.name != "dummy":
            self.job_dags.remove(job_dag)
            self.finished_job_dags.add(job_dag)
        self.action_map = compute_act_map(self.job_dags)

    def reset(self, max_time=np.inf):
        self.total_carbon_usage = 0
        self.last_time = 0
        self.max_time = max_time
        self.wall_time.reset()
        self.timeline.reset()
        self.exec_commit.reset()
        self.moving_executors.reset()
        self.reward_calculator.reset()
        self.finished_job_dags = OrderedSet()
        self.node_selected.clear()
        self.last_CI_time = 0
        for executor in self.executors:
            executor.reset()
        self.free_executors.reset(self.executors)
        # generate an empty set of job_dags
        self.job_dags = OrderedSet()
        # map action to dag_idx and node_idx
        self.action_map = compute_act_map(self.job_dags)
        # add initial set of jobs in the system
        for job_dag in self.job_dags:
            self.add_job(job_dag)
        # put all executors as source executors initially
        self.source_job = None
        self.num_source_exec = len(self.executors)
        self.exec_to_schedule = OrderedSet(self.executors)

    def seed(self, seed):
        self.np_random.seed(seed)