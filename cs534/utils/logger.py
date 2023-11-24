import os
import csv

class Logger(object):
    ''' Logger saves the running results and helps make plots from the results
    '''

    def __init__(self, log_dir):
        ''' Initialize the labels, legend and paths of the plot and log file.

        Args:
            log_path (str): The path the log files
        '''
        self.log_dir = log_dir

    def __enter__(self):
        self.txt_path = os.path.join(self.log_dir, 'log.txt')
        self.csv_path = os.path.join(self.log_dir, 'performance.csv')
        self.fig_path = os.path.join(self.log_dir, 'fig.png')
        self.stored_values = []

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.txt_file = open(self.txt_path, 'a')
        self.csv_file = open(self.csv_path, 'a')
        fieldnames = ['episode', 'reward', 'random_reward']
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        if (os.stat(self.csv_path).st_size == 0):
            self.writer.writeheader()

        return self

    def log(self, text):
        ''' Write the text to log file then print it.
        Args:
            text(string): text to log
        '''
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def store_performance(self, episode, rule_reward, random_reward):
        ''' Log a point in the curve
        Args:
            episode (int): the episode of the current point
            reward (float): the reward of the current point
            random_reward (float): the reward if matched against a random agent
        '''
        self.stored_values.append((episode, rule_reward, random_reward))
        print('')
        self.log('----------------------------------------')
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(rule_reward))
        self.log('  vs random    |  ' + str(random_reward))
        self.log('----------------------------------------')

    def flush_performance(self):
        for episode, rule_reward, random_reward in self.stored_values:
            self.writer.writerow({'episode': episode, 'reward': rule_reward, 'random_reward': random_reward})
        self.stored_values = []
        self.csv_file.flush()

    def __exit__(self, type, value, traceback):
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()
        print('\nLogs saved in', self.log_dir)

def plot_curve(csv_path, save_path, algorithm):
    ''' Read data from csv file and plot the results
    '''
    import os
    import csv
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        episodes = []
        rewards = []
        random_rewards = []

        for row in reader:
            episodes.append(int(row['episode']))
            rewards.append(float(row['reward']))
            random_rewards.append(float(row['random_reward']))
        fig, ax = plt.subplots()
        ax.plot(episodes, rewards, label=f"{algorithm} vs rule")
        ax.plot(episodes, random_rewards, label=f"{algorithm} vs random")
        ax.set(xlabel='episode', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)
