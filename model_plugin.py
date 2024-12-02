import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from spark_env.env import Environment
import time
import yaml
import pandas as pd
from spark_env.job_dag import JobDAG
from spark_env.node import Node
from spark_env.task import Task  # Assuming you have a Task class
from spark_env.wall_time import WallTime  # Assuming you have a WallTime class
from spark_env.executor_commit import ExecutorCommit
from spark_env.moving_executors import MovingExecutors
from collections import OrderedDict
from spark_env.action_map import compute_act_map
from utils import *
from spark_env.executor import Executor
from actor_agent import ActorAgent
from danish_actor_agent import DANISHAgent
import time
import subprocess

MAX_EXECS = 4
PATH_TO_RESOURCE_QUOTA = "../cap-k8s/resource_quota.yaml"
executors = [Executor(exec_id) for exec_id in range(MAX_EXECS)]
sess = None
agent = None
agent_name = None
wall_time = None
cap_agent = None

def init(name):
    global sess, agent, agent_name, wall_time, cap_agent
    if cap_agent is not None:
        # kill the subprocess
        cap_agent.kill()
    # set the maximum resource quota to start
    update_resource_quota()
    # initialize WallTime
    wall_time = WallTime()
    # reset tensorflow
    tf.compat.v1.reset_default_graph() 
    tf.compat.v1.set_random_seed(42)
    sess = tf.compat.v1.Session()
    if name == "decima":
        agent = DANISHAgent(sess, 5, 3, [16, 8], 8, 8, range(1, MAX_EXECS + 1), {0.0: 150}, gamma=0.0)
        agent_name = name
    elif name == "danish":
        agent = DANISHAgent(sess, 5, 3, [16, 8], 8, 8, range(1, MAX_EXECS + 1), {0.0: 0})
        agent_name = name
    elif name == "default":
        agent = DANISHAgent(sess, 5, 3, [16, 8], 8, 8, range(1, MAX_EXECS + 1), {0.0: 0})
        agent_name = "default"
    elif name == "cap":
        agent = DANISHAgent(sess, 5, 3, [16, 8], 8, 8, range(1, MAX_EXECS + 1), {0.0: 0})
        agent_name = "cap"
        # if the agent is cap, run the cap agent in the background
        print("Running CAP agent in the background")
        cap_agent = run_cap_agent()

# runs the cap agent in background if we're using CAP
def run_cap_agent():
    # run the command "python3 ~/GitHub/cap-k8s/cap.py --namespace spark-ns --res-quota-path resource_quota.yaml --api-domain 127.0.0.1:6066 --min-execs 4 --max-execs 10 --interval 60"
    # in the background
    return subprocess.Popen(["python3", "../cap-k8s/cap.py", "--namespace", "spark-ns", "--res-quota-path", "../cap-k8s/resource_quota.yaml", "--api-domain", "127.0.0.1:6066", "--min-execs", "4", "--max-execs", "10", "--interval", "60"])

# Update the Kubernetes resource quota
def update_resource_quota():
    try:
        # Load the existing resource quota definition from the YAML file
        with open(PATH_TO_RESOURCE_QUOTA, "r") as f:
            resource_quota = yaml.safe_load(f)

        # Update the allowable pods
        resource_quota["spec"]["hard"]["cpu"] = str(12) # 1 cpu core per exec
        resource_quota["spec"]["hard"]["memory"] = str(24) + "Gi" # 1Gi memory per exec

        # Save the updated resource quota definition back to the YAML file
        with open(PATH_TO_RESOURCE_QUOTA, "w") as f:
            yaml.safe_dump(resource_quota, f, default_flow_style=False)

        # Apply the resource quota using kubectl
        subprocess.run(
            ["kubectl", "apply", "-f", PATH_TO_RESOURCE_QUOTA, "-n", "spark-ns"],
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error updating resource quota: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def reconstruct_job_dags(app_registry, source_job_input, frontier_nodes_input, exec_commit_input, moving_executors_input, num_source_exec):
    global wall_time
    job_dags = []
    executor_limits = {}
    moving_nodes = []
    frontier_nodes = OrderedSet()
    source_job = None
    exec_commit = ExecutorCommit()
    moving_executors = MovingExecutors()
    node_to_jobstage_map = {}
    
    # all dummy values for now
    np_random = np.random.RandomState()

    for app_id in app_registry.keys():
        if "driver_pod" not in app_registry[app_id].keys():
            continue
        for job_id in app_registry[app_id].keys():
            if job_id in ["exec_limit", "driver_pod"]:
                continue
            stages = app_registry[app_id][job_id]
            # print(stages)
            nodes = []
            adj_mat = np.zeros((len(stages), len(stages)), dtype=int)
            stage_id_to_idx = {}
            cur_idx = 0
            for stage_id in stages.keys():
                stage_id_to_idx[stage_id] = cur_idx
                cur_idx += 1
            for stage_id in stages.keys():
                stage_info = stages[stage_id]
                num_tasks = stage_info["num_tasks"] if "num_tasks" in stage_info.keys() else 1
                finished_tasks = stage_info["num_finished_tasks"] if "num_finished_tasks" in stage_info.keys() else 1

                # create a list of task instances
                tasks = [Task(i, 0.1, wall_time) for i in range(num_tasks)]
                # and task durations
                task_duration = {
                    'first_wave': {i: [1,2,3] for i in range(num_tasks)},  # Example values
                    'rest_wave': {i: [0,0,0] for i in range(num_tasks)},  # Example values
                    'fresh_durations': {i: [0,0,0] for i in range(num_tasks)}  # Example values
                }

                node = Node(stage_id_to_idx[stage_id], tasks, task_duration, wall_time, np_random)
                node.set_finished_tasks(finished_tasks)
                nodes.append(node)

                # map each node to an app_id, job_id, and stage_id
                node_to_jobstage_map[node] = (app_id, job_id, stage_id)

                if app_id in frontier_nodes_input.keys() and job_id in frontier_nodes_input[app_id].keys() and stage_id in frontier_nodes_input[app_id][job_id]:
                    frontier_nodes.add(node)

                if stages[stage_id]["status"] == "COMPLETE":
                    moving_nodes.append(node)

                for dep_id in stage_info["dependencies"]:
                    if dep_id not in stage_id_to_idx:
                        continue
                    adj_mat[stage_id_to_idx[dep_id], stage_id_to_idx[stage_id]] = 1
            job_dag = JobDAG(nodes, adj_mat, f"{app_id}_job_{job_id}")
            job_dags.append(job_dag)

            executor_limits[job_dag] = app_registry[app_id]["exec_limit"]

            # set the source job if the source job is in the current app_id
            if source_job_input == app_registry[app_id]["driver_pod"]:
                # if all of the stages in the current job_id are "SKIPPED" or "COMPLETE", this is the source job
                # look at the stages in app_registry[app_id][job_id]
                if all([stages[stage_id]["status"] in ["SKIPPED", "COMPLETE"] for stage_id in stages.keys()]):
                    source_job = job_dag

            # set the executor commits
            for _, job_id in exec_commit_input:
                if job_id == app_registry[app_id]["driver_pod"]:
                    exec_commit.add_job(job_dag)
            
            # set the moving executors
            for i in range(moving_executors_input):
                if len(moving_nodes) > 0:
                    # choose a node from moving_nodes (randomly for now)
                    node = moving_nodes[np_random.randint(len(moving_nodes))]
                    moving_executors.add(executors[i], node)

    return node_to_jobstage_map, job_dags, num_source_exec, executor_limits, frontier_nodes, source_job, exec_commit, moving_executors, compute_act_map(job_dags), node_to_jobstage_map

# get actions
def get_actions(app_registry, source_job_input, frontier_nodes_input, exec_commit_input, moving_executors_input, num_source_exec, carbon_intensity, L, U):
    global agent, wall_time

    # get current time and update wall_time
    new_time = time.time()
    wall_time.update_time(new_time)
    
    node_to_jobstage_map, job_dags, num_source_exec, exec_limits, frontier_nodes, source_job, exec_commit, moving_executors, action_map, node_to_jobstage_map = reconstruct_job_dags(app_registry, source_job_input, frontier_nodes_input, exec_commit_input, moving_executors_input, num_source_exec)
    obs = job_dags, source_job, num_source_exec, frontier_nodes, exec_limits, exec_commit, moving_executors, action_map, new_time
    
    # set the carbon intensity for the agent
    if agent_name == "danish":
        carbon_schedule = {new_time-100: carbon_intensity}
        agent.set_carbon_schedule(carbon_schedule, L, U)

    node, use_exec, node_act_probs, job_act_probs = agent.get_action(obs)

    # use node act probs to provide weights to each job (each unique app_id and job_id pair)
    job_prob = {}
    stage_prob = {}

    for i, prob in enumerate(node_act_probs[0]):
        node = action_map[i]
        app_id, job_id, stage_id = node_to_jobstage_map[node]
        if app_id not in job_prob.keys():
            job_prob[app_id] = {}
            stage_prob[app_id] = {}
        if job_id not in job_prob[app_id].keys():
            job_prob[app_id][job_id] = 0.0
        job_prob[app_id][job_id] += prob
        stage_prob[app_id][stage_id] = prob
    
    sorted_app_id = {k: sum(v.values()) for k, v in job_prob.items()}
    sorted_app_id = {k: v for k, v in sorted(sorted_app_id.items(), key=lambda item: item[1], reverse=True)}

    sorted_stage_prob = {app_id: {k: v for k, v in sorted(stage_prob[app_id].items(), key=lambda item: item[1], reverse=True)} for app_id in stage_prob.keys()}

    # get carbon aware
    carbonAware = False
    if agent_name == "danish" and use_exec == 0:
        carbonAware = True
    
    if node is None:
        return None, carbonAware, sorted_app_id, sorted_stage_prob

    # get actions
    return node_to_jobstage_map[node], carbonAware, sorted_app_id, sorted_stage_prob