
# coding: utf-8

# In[ ]:

import numpy as np
import pickle as pickle
import tensorflow as tf
#%matplotlib inline
import matplotlib.pyplot as plt
import math
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import pandas as pd
df = pd.read_csv('RB00.10s.ws.csv.gz')

I = 0
R = 0
V = 0 
R_max = 0
Cir = 0
def envreset():
    global I, R, V, R_max,Cir
    I = 0
    R = 0
    V = 0 
    R_max = 0
    Cir = 0
    return df.iloc[0,1:-1]

def envstep(dual):
    global I, R, V , R_max ,Cir
    dual = 1 if dual==1 else -1
    done = 0 
    #print(I, R, V, done, dual)
    observation = df.iloc[I,1:-1]
    reward = df.iloc[I,-1]
    #reward = 5 if reward > 5 else (-5 if reward < -5 else reward)
    if V == 0:
        R = R + reward*dual
        V = dual 
    elif V ==1:
        if dual == -1:
            R = R + 2*reward*dual
            V = -1
        if dual ==1:
            R = R + reward*dual
            #V = 1
    elif V ==-1:
        if dual == 1:
            R = R + 2*reward*dual
            V = 1
        if dual ==-1:
            R = R + reward*dual
            #V = -1
        
    if R>= R_max:
        R_max = R
        
    I =I + 1
    if I>= len(df)-1:
        I=0
        Cir =Cir+1
    if R/(Cir*len(df)+(I+1)) <0.0001:
        done = 1
    Reward = 1.0 if (R/(I+1)>=0.0001) else 0.0
    #print(dual*reward)
    return observation, Reward, done, reward
    

H = 500 
batch_size = 1 # every how many episodes to do a param update?
learning_rate = 1e-4 # feel free to play with this to train faster or more stably.
gamma = 0.99 # discount factor for reward

D = 240 # input dimensionality





# In[ ]:


tf.reset_default_graph()

#This defines the network as it goes from taking an observation of the environment to 
#giving a probability of chosing to the action of moving left or right.
observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
probability = tf.nn.sigmoid(score)

#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

# The loss function. This sends the weights in the direction of making actions 
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
#lik = (input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages) 
newGrads = tf.gradients(loss,tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))



def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init =tf.global_variables_initializer()
pred = []
label = []
# Launch the graph
with tf.Session() as sess:
    #rendering = False
    sess.run(init)
    observation = envreset() # Obtain an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in 
    # gradBuffer until we are ready to update our policy network. 
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    
    while episode_number <= total_episodes:
        
        # Rendering the environment slows things down, 
        # so let's only look at it once our agent is doing a good job.
        #if reward_sum/batch_size > 100 or rendering == True : 
            #env.render()
            #rendering = True
            
        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation,[1,D])

        # Run the policy network and get an action to take. 
        tfprob = sess.run(probability,feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        #print(tfprob,action)
        
        xs.append(x) # observation
        y = 1 if action == 0 else 0 # a "fake label"
        ys.append(y)

        # step the environment and get new measurements
        observation, reward, done, lbl= envstep(action)
        #print(pred, action)
        pred.append(action)
        label.append(lbl)
        #print(reward,reward_sum)
        reward_sum += reward
        

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: 
            #print("done1")
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            #discounted_epr /= np.std(discounted_epr)
            #print("loglik:",sess.run(loglik))
            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad
                
            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0: 
                #break
                print(R_max)
                if reward_sum/batch_size>20000:
                    print("reward_average>20000")
                    print(pred)
                    print(label)
                sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                
                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print("Average reward for episode %f.  Total average reward %f." % (reward_sum/batch_size, running_reward/batch_size))
                
                if reward_sum/batch_size >= 20000: 
                    print("Task solved in",episode_number,"episodes!")     
                    print(pred)
                    print(label)
                    break
                    
                reward_sum = 0 
            
            observation = envreset()
            
            pred = []
            label = []
        
print(episode_number,"Episodes completed.")


# In[ ]:

gradBuffer


# In[ ]:

tvars


# In[ ]:

tGrad


# In[ ]:

print(len(pred),len(label))


# In[ ]:

R_new = 0
for i in range(len(label)):
    if pred[i]==1:
        R_new =R_new+label[i]
    else:
        R_new =R_new - label[i]
        
        
print(R_new)


# In[ ]:

R_max


# In[ ]:

R


# In[ ]:

2.5*40*7


# In[ ]:

aaa = [1,3,4,5]
bbb = [2,4,6,8]
np.mat(aaa).T*np.mat(bbb)


# In[ ]:



