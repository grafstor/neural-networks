#-------------------------------#
#       Author: grafstor        
#       Date: 06.09.20          
#-------------------------------#

from time import sleep
import random

from graphics import *
from geras import *

SHOW = True

win_s = 20

if SHOW:
    win = GraphWin("SNAKE", win_s*30, win_s*30)

class Snake:
    def __init__(self):
        self.draw = False
        self.reset()

    def reset(self):
        self.head_x = 10
        self.head_y = 10

        self.body_x = [0]
        self.body_y = [0]

        self.is_alive = True

        if SHOW:
            if self.draw:
                self.head.undraw()
                for part in self.body:
                    part.undraw()
            self.draw = True

            self.body = []
            self.head = Rectangle(Point(0,0), Point(30,30))
            self.head.setOutline("red")
            self.head.setFill("red")
            self.head.draw(win)
            self.head.move(self.head_x*30, self.head_y*30)

        self.body_x.append(self.head_x)
        self.body_y.append(self.head_y)

    def move_head(self,x,y):
        self.head_x += x
        self.head_y += y

        if self.head_x < 0:
            self.is_alive = False
        elif self.head_y < 0:
            self.is_alive = False
        elif self.head_x >= win_s-1:
            self.is_alive = False
        elif self.head_y >= win_s-1:
            self.is_alive = False

        for i in range(len(self.body_x)-2,-1,-1):
            self.body_x[i+1] = self.body_x[i]
            self.body_y[i+1] = self.body_y[i]
        
        if SHOW:
            for i in range(len(self.body)):
                self.body[i].move(self.body_x[i+1]*30 - self.body_x[i+2]*30,
                                    self.body_y[i+1]*30 - self.body_y[i+2]*30)
            self.head.move(x*30,y*30)

        for i in range(len(self.body_x)-2):
            if self.body_x[i+1] == self.head_x:
                if self.body_y[i+1] == self.head_y:
                    self.is_alive = False

        self.body_x[0] = self.head_x
        self.body_y[0] = self.head_y

    def new_part(self):
        self.body_x.append(0)
        self.body_y.append(0)

        if SHOW:
            self.body.append(Rectangle(Point(self.body_x[-2]*30,self.body_y[-2]*30),
                                    Point(self.body_x[-2]*30+30,self.body_y[-2]*30+30)))
            self.body[-1].setOutline("red")
            self.body[-1].setFill("red")
            self.body[-1].draw(win)

    def head_coord(self):
        return (self.head_x, self.head_y)


class Apple:
    def __init__(self):
        self.draw = False
        self.reset()

    def reset(self):
        self.x = random.randint(0,win_s-1) 
        self.y = random.randint(0,win_s-1) 

        if SHOW:
            if self.draw:
                self.apple.undraw()
            self.draw = True
            self.apple = Circle(Point(self.x*30 + 15,self.y*30 + 15), 15).draw(win)
            self.apple.setOutline("green")
            self.apple.setFill("green")

    def coord(self):
        return (self.x, self.y)


class Game:
    def __init__(self):
        self.snake = Snake()
        self.apple = Apple()
        self.last_action = 4

    def step(self, action):
        done = False
        reward = 0
        action += 1

        self.last_action = action

        if action == 1: 
            self.snake.move_head(0,-1)
        elif action == 2:
            self.snake.move_head(0,1)
        elif action == 4:
            self.snake.move_head(1,0)
        elif action == 3:
            self.snake.move_head(-1,0)

        if self.apple.coord() == self.snake.head_coord():
            self.apple.reset()
            self.snake.new_part()
            reward = 1

        if not self.snake.is_alive:
            done = True

        state = self.__get_state()

        return (state, reward, done)

    def reset(self):
        self.snake.reset()
        self.apple.reset()
        return self.__get_state()

    def __get_state(self):
        state = [0 for i in range(win_s*win_s*2+4)]

        for i in range(len(self.snake.body_x)-1):
            for j in range(len(self.snake.body_y)-1):
                state[self.snake.body_x[i]*win_s + self.snake.body_y[j]] = 1

        head_coord = self.snake.head_coord()
        coord = self.apple.coord()

        state[head_coord[0]*win_s + head_coord[1]] = 1
        state[win_s*win_s+coord[0]*win_s + coord[1]] = 1

        for i in range(4):
            if i+1 == self.last_action:
                state[win_s*win_s*2+i] = 1

        state = np.array(state)

        return state


class Session:
    def __init__(self, model, epsilon=0.9, gamma=0.8, decay_rate=0.005, min_epsilon=0.1):
        self.model = model 

        self.epsilon = epsilon 
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

        self.memory_size = 300
        self.memory = []

        self.env = Game()
        self.n_states = win_s*win_s*2+4
        self.n_actions = 4

    def __memorize(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def __construct_training_set(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])

        Q = self.model.predict(states)
        Q_new = self.model.predict(new_states)

        replay_size = len(replay)

        X = np.empty((replay_size, self.n_states))
        y = np.empty((replay_size, self.n_actions))
        
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, done_r = replay[i]

            target = Q[i]
            target[action_r] = reward_r

            if not done_r:
                target[action_r] += self.gamma * np.amax(Q_new[i])

            X[i] = state_r
            y[i] = target

        return X, y

    def __select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)

        else:
            action = np.argmax(self.model.predict(state), axis=1)[0]

        return action

    def train(self, n_epochs=5000, batch_size=32):
        max_reward = 0

        for epoch in range(n_epochs):
            state = self.env.reset()
            total_reward = 0

            epoch_loss = []
            while True:
                action = self.__select_action(state)
                new_state, reward, done = self.env.step(action)

                self.__memorize(state, action, reward, new_state, done)

                _batch_size = min(len(self.memory), batch_size)
                replay = random.sample(self.memory, _batch_size)

                X, y = self.__construct_training_set(replay)

                prediction = self.model.train(X, y)
                loss =  np.mean(self.model.loss.loss(y, prediction))
                epoch_loss.append(loss)

                total_reward += reward
                state = new_state

                if done:
                    break

            epoch_loss = np.array(epoch_loss)
            epoch_loss = np.mean(epoch_loss,  keepdims = False)

            self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-self.decay_rate * epoch)
            
            max_reward = max(max_reward, total_reward)

            print ("%d [Loss: %.4f, Reward: %s, Epsilon: %.4f, Max Reward: %s]" % (epoch, epoch_loss, total_reward, self.epsilon, max_reward))


def main():
    model = Model(
        Dense(300),
        Sigmoid(),

        Dense(100),
        Sigmoid(),

        Dense(4),
        Softmax(), 

    )(Adam(0.001))

    game = Session(model)
    game.train()

if __name__ == '__main__':
    main()
