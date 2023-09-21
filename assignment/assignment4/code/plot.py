from matplotlib import pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alpha1', default=0.01, type=float, help='alpha in DQN')
parser.add_argument('--alpha2', default=0.5, type=float, help='alpha in DQN')
parser.add_argument('-r','--reward', default=1, type=int, help='Reward type')
parser.add_argument('-n','--network', default=1, type=int, help='Network type')
parser.add_argument('-m','--mode', default=1, type=int, help='Plotting mode, mode1 for DQN and DDQN, mode2 for Dueling DQN')
args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == 1:
        name = '_network'+ str(args.network) + '_reward' + str(args.reward)
        DQN_name = str(args.alpha1) + name
        DDQN_name = str(args.alpha2) + name
        DQN_loss_file = 'log/DQN_alpha' + DQN_name + '_loss_per_episode.npy'
        DDQN_loss_file = 'log/Double_DQN_alpha' + DDQN_name + '_loss_per_episode.npy'
        DQN_step_file = 'log/DQN_alpha' + DQN_name + '_step_per_episode.npy'
        DDQN_step_file = 'log/Double_DQN_alpha' + DDQN_name + '_step_per_episode.npy'
        DQN_time_file = 'log/DQN_alpha' + DQN_name + '_time_per_episode.npy'
        DDQN_time_file = 'log/Double_DQN_alpha' + DDQN_name + '_time_per_episode.npy'

        DQN_loss = np.load(DQN_loss_file)
        DDQN_loss = np.load(DDQN_loss_file)
        DQN_step = np.load(DQN_step_file)
        DDQN_step = np.load(DDQN_step_file)
        DQN_time = np.load(DQN_time_file)
        DDQN_time = np.load(DDQN_time_file)

        loss_fig = 'com_figure/{}_loss.png'.format(name)
        step_fig = 'com_figure/{}_step.png'.format(name)
        time_fig = 'com_figure/{}_time.png'.format(name)

        step_list_x = np.arange(len(DQN_loss))
        plt.plot(step_list_x, DQN_loss, color="tab:blue")
        plt.plot(step_list_x, DDQN_loss, color="tab:orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss of episode")
        plt.legend(["DQN", "Double DQN"])
        plt.savefig(loss_fig)

        step_list_x = np.arange(len(DQN_step))
        plt.cla()
        plt.plot(step_list_x, DQN_step, color="tab:blue")
        plt.plot(step_list_x, DDQN_step, color="tab:orange")
        plt.xlabel("Epochs")
        plt.ylabel("Step of episode")
        plt.legend(["DQN", "Double DQN"])
        plt.savefig(step_fig)

        step_list_x = np.arange(len(DQN_time))
        plt.cla()
        plt.plot(step_list_x, DQN_time, color="tab:blue")
        plt.plot(step_list_x, DDQN_time, color="tab:orange")
        plt.xlabel("Epochs")
        plt.ylabel("Time of episode")
        plt.legend(["DQN", "Double DQN"])
        plt.savefig(time_fig)

    else:
        name = str(args.alpha1) + '_network'+ str(args.network) + '_reward' + str(args.reward)
        loss_file = 'log/Dueling_DQN_alpha' + name + '_loss_per_episode.npy'
        step_file = 'log/Dueling_DQN_alpha' + name + '_step_per_episode.npy'
        time_file = 'log/Dueling_DQN_alpha' + name + '_time_per_episode.npy'

        loss = np.load(loss_file)
        step = np.load(step_file)
        time = np.load(time_file)

        print(loss.shape)
        print(step.shape)
        print(step.shape)

        loss_fig = 'com_figure/{}_loss.png'.format(name)
        step_fig = 'com_figure/{}_step.png'.format(name)
        time_fig = 'com_figure/{}_time.png'.format(name)

        step_list_x = np.arange(len(loss))
        plt.plot(step_list_x, loss, color="tab:blue")
        plt.xlabel("Epochs")
        plt.ylabel("Loss of episode")
        plt.savefig(loss_fig)

        step_list_x = np.arange(len(step))
        plt.cla()
        plt.plot(step_list_x, step, color="tab:blue")
        plt.xlabel("Epochs")
        plt.ylabel("Step of episode")
        plt.savefig(step_fig)

        step_list_x = np.arange(len(time))
        plt.cla()
        plt.plot(step_list_x, time, color="tab:blue")
        plt.xlabel("Epochs")
        plt.ylabel("Time of episode")
        plt.savefig(time_fig)
