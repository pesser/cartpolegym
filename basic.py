def getting_started():
    """First steps in running an OpenAI Gym environment."""
    # available on pip as
    # pip install gym
    import gym
    import time

    # loading an environment
    env = gym.make('CartPole-v0')

    # typically we run multiple episodes of an environment
    for i_episode in range(20):
        # prepare environment for new episode and return the initial state
        # observation
        observation = env.reset()
        # an episode runs in discrete time steps
        for t in range(100):
            # slow down fps for visualization
            time.sleep(0.01)
            # render will open/update a window with a visualization of the
            # environment and its current state
            env.render()
            # take a look at what an observation looks like
            print(observation)
            # to continue with the simulation, we need to provide an action.
            # just like the observation, the actions depend on the specific
            # environment. for convenience we can just obtain a randomly choosen
            # action:
            action = env.action_space.sample()
            # step continues the simulation using the provided action and
            # returns a new observation of the state, a reward and a boolean
            # indicating whether this episode is done
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

def observations_and_actions():
    """A closer look at the main objects used to interact with
    environments."""
    import gym
    import time

    env = gym.make("CartPole-v0")
    # Possible values for observations and actions can be obtained from the
    # corresponding space attributes:
    print(env.observation_space)
    # > Box(4,)
    print(env.action_space)
    # > Discrete(2)
    # A Discrete space specifies the integers from zero to the (exclusive)
    # upper bound specified, i.e. the integers zero and one in this case
    # A Box space specifies points in a N-dimensional box with each
    # dimension constraint to the interval specified in
    print(env.observation_space.low)
    # > [ -4.80000000e+00  -3.40282347e+38  -4.18879020e-01  -3.40282347e+38]
    print(env.observation_space.high)
    # > [  4.80000000e+00   3.40282347e+38   4.18879020e-01   3.40282347e+38]
    # while our learning algorithm should figure out which actions to take,
    # it might nevertheless be good to know what the actions are doing:
    for action in [0,1]:
        env.reset()
        for t in range(100):
            time.sleep(0.1)
            env.render()
            _,_,done,_ = env.step(action)
            if done:
                print("Finished episode with constant action {}".format(action))
                break
    # We see that 0 corresponds to "move left" and 1 corresponds to "move
    # right"


def learn():
    """Where we learn to balance. Based on
    https://gist.github.com/Irlyue/678de7fce2cc2c72e71215890f547086#file-cart-pole-py
    """
    import gym
    import tensorflow as tf
    import numpy as np


    def discounted_reward(rewards, gamma):
        """Compute the discounted reward."""
        ans = np.zeros_like(rewards)
        running_sum = 0
        # compute the result backward
        for i in reversed(range(len(rewards))):
            running_sum = running_sum * gamma + rewards[i]
            ans[i] = running_sum
        return ans


    def log_prob_from_logits(logits):
        """Log probability of pdf of categorical distribution corresponding
        to logits."""
        return logits - tf.reduce_logsumexp(logits, axis = 1, keep_dims = True)
    

    class Model(object):
        def __init__(self, session):
            # forward pass from state to sampled action
            # 4 dimensional observation
            state = tf.placeholder(tf.float32, [None, 4])
            # two layer fully connected to produce logits on action space
            h = state
            h = tf.layers.dense(h, 32, activation = tf.nn.relu)
            action_logits = tf.layers.dense(h, 2, activation = None)
            # in this case a single output would have been enough but
            # we implement it such that it readily generalizes to
            # larger action spaces
            # sample action via gumbel max
            uniform = tf.random_uniform(tf.shape(action_logits), minval = 1e-6, maxval = 1 - 1e-6)
            gumbel = -tf.log(-tf.log(uniform))
            action_sample = tf.argmax(action_logits + gumbel, axis = 1)

            # forward pass to action sample is computed in each step. The
            # state, sampled action and subsequently received reward are remembered in
            # each step. At the end of the episode, we recompute the forward
            # passes and compute the backward passes to obtain
            # the gradients of the loss which is derived from the sampled
            # action and the corresponding reward received.

            action = tf.placeholder(tf.int32, [None])
            reward = tf.placeholder(tf.float32, [None])
            # get sensitivities of responsible logit
            log_probs = tf.reduce_sum(
                    log_prob_from_logits(action_logits) * tf.one_hot(action, 2),
                    axis = 1)
            # increase logits with positive rewards, decrease logits with
            # negative rewards
            loss = -tf.reduce_mean(log_probs * reward)

            # optimization
            optimizer = tf.train.AdamOptimizer(learning_rate = 1e-2)
            optimize = optimizer.minimize(loss)

            # interface
            ## inputs
            self.state = state
            self.action = action
            self.reward = reward
            ## outputs
            self.action_sample = action_sample
            ## opts
            self.optimize = optimize

            # initialize
            self.session = session
            self.session.run(tf.global_variables_initializer())
            self.history = {
                    "state": [],
                    "action": [],
                    "reward": []}


        def sample_action(self, state):
            """Sample action from policy conditioned on state."""
            action = self.session.run(self.action_sample, {
                self.state: [state]})
            action = action[0]
            self.history["state"].append(state)
            self.history["action"].append(action)
            return action


        def receive_reward(self, reward):
            """Remember reward for our actions."""
            self.history["reward"].append(reward)
            # consistency check
            n = len(self.history["reward"])
            assert len(self.history["state"]) == n
            assert len(self.history["action"]) == n


        def update(self):
            """Update variables at end of episode."""
            rewards = self.history["reward"]
            rewards = discounted_reward(rewards, 0.95)
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            self.history["reward"] = rewards
            self.session.run(self.optimize, {
                self.state: self.history["state"],
                self.action: self.history["action"],
                self.reward: self.history["reward"]})
            self.history = {
                    "state": [],
                    "action": [],
                    "reward": []}
            

    env = gym.make('CartPole-v0')
    session = tf.Session()
    model = Model(session)
    for i_episode in range(1000):
        observation = env.reset()
        total_reward = 0.0
        while True:
            action = model.sample_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            model.receive_reward(reward)
            env.render()
            if done:
                break
        model.update()
        if i_episode % 50 == 0:
            print("episode {}, reward {}".format(i_episode, total_reward))



if __name__ == "__main__":
    #getting_started()
    #observations_and_actions()
    learn()
