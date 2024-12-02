import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from spark_env.env import Environment
from spark_agent import SparkAgent
from heuristic_agent import DynamicPartitionAgent
from actor_agent import ActorAgent
from danish_actor_agent import DANISHAgent
from carbon_aware_agent import CarbonAgent
from spark_env.canvas import *
from param import *
from utils import *
import time
import pandas as pd

# create result folder
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

# tensorflo seeding
tf.compat.v1.set_random_seed(args.seed)

df = pd.read_csv("us-west-2.csv")
c = df["carbon_intensity_avg"]
carbon_schedule = [(1000* i, c[i]) for i in range(len(c.to_list()))]

carbon_dict = {}
for i in range(len(c.to_list())):
    carbon_dict[1000*i] = c[i]

# set up environment
env = Environment(carbon_schedule=carbon_dict)

# set up agents
agents = {}
carbon_time_list = []

for scheme in args.test_schemes:
    if scheme == 'learn':
        sess = tf.compat.v1.Session()
        agents[scheme] = ActorAgent(
            sess, args.node_input_dim, args.job_input_dim,
            args.hid_dims, args.output_dim, args.max_depth,
            range(1, args.exec_cap + 1))
    elif scheme == 'danish':
        agents[scheme] = None
    elif scheme == 'dynamic_partition':
        agents[scheme] = DynamicPartitionAgent()
    elif scheme == 'spark_fifo':
        agents[scheme] = SparkAgent(exec_cap=args.exec_cap)
    elif scheme == 'cap':
        agents[scheme] = CarbonAgent(exec_cap=args.exec_cap, carbon_schedule=carbon_dict)
    else:
        print('scheme ' + str(scheme) + ' not recognized')
        exit(1)

# store info for all schemes
all_total_reward = {}
for scheme in args.test_schemes:
    all_total_reward[scheme] = []


for exp in range(args.num_exp):
    print('Experiment ' + str(exp + 1) + ' of ' + str(args.num_exp))

    for scheme in args.test_schemes:
        print('Scheme ' + scheme)
        # reset environment with seed
        env.seed(args.num_ep + exp)
        env.reset()

        # load an agent
        agent = agents[scheme]

        # start experiment
        obs = env.observe()
        # print("obs", obs) # job_dags, source_job, num_source_exec, get_frontier_nodes(), get_executor_limits(), exec_commit, moving_executors, action_map, wall_time.curr_time
        print("job_dags", obs[0])
        print("source_job", obs[1])
        print("num_source_exec", obs[2])
        print("get_frontier_nodes()", obs[3])
        print("get_executor_limits()", obs[4])
        print("exec_commit", obs[5])
        print("moving_executors", obs[6])
        print("action_map", obs[7])
        print("wall_time.curr_time", obs[8])

        time.sleep(10)
        # print("env", env)

        total_reward = 0
        done = False
        i = 0
        if scheme != 'danish':
            while not done:
                # print a single dot every 10 steps to indicate progress (all on same line)
                if i % 10 == 0:
                    print('.', end='', flush=True)
                i += 1
                node, use_exec = agent.get_action(obs)
                obs, reward, done = env.step(node, use_exec)
                total_reward += reward
        else: 
            # refresh tensorflow completely
            tf.compat.v1.reset_default_graph() 
            tf.compat.v1.set_random_seed(args.seed)
            sess = tf.compat.v1.Session()
            agent = DANISHAgent(
                sess, args.node_input_dim, args.job_input_dim,
                args.hid_dims, args.output_dim, args.max_depth,
                range(1, args.exec_cap + 1), carbon_dict)
            while not done:
                node, use_exec, cw = agent.get_action(obs)
                obs, reward, done = env.step(node, use_exec, carbon_aware = cw)
                total_reward += reward

        all_total_reward[scheme].append(total_reward)

        if args.canvs_visualization == 0: # this version of visualization needs update for new carbon accounting
            total_time = env.wall_time.curr_time
            total_carbon = env.total_carbon_usage
            carbon_time_list.append((scheme, total_time/1000, total_carbon))  
            

        elif args.canvs_visualization == 1:
            visualize_dag_time_save_pdf(
                env.finished_job_dags, env.executors,
                args.result_folder + 'visualization_exp_' + \
                str(exp) + '_scheme_' + scheme + \
                '.png', plot_type='app')
        elif args.canvs_visualization == 2:
            visualize_executor_usage(env.finished_job_dags,
                args.result_folder + 'visualization_exp_' + \
                str(exp) + '_scheme_' + scheme + '.png', carbon_dict)


    # plot CDF of performance
    if args.canvs_visualization == 0:
        visualize_carbon(carbon_time_list)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for scheme in args.test_schemes:
        x, y = compute_CDF(all_total_reward[scheme])
        ax.plot(x, y)

    plt.xlabel('Total reward')
    plt.ylabel('CDF')
    plt.legend(args.test_schemes)
    fig.savefig(args.result_folder + 'total_reward.png')

    plt.close(fig)
