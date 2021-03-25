using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//
using SixLabors.ImageSharp;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;
//
using NumSharp;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using CustomRandom;
//

// Initial framework taken // from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

// import numpy as np

// import gym

// from keras.models import Model
// from keras.layers import Input, Dense
// from keras import backend as K
// from keras.optimizers import Adam

// import numba as nb
// from tensorboardX import SummaryWriter

var ENV = "LunarLander-v2";
var CONTINUOUS = false;

var EPISODES = 100000;

var LOSS_CLIPPING = 0.2; // Only implemented clipping for (int the surrogate loss, paper said it was best
var EPOCHS = 10;
var NOISE = 1.0; // Exploration noise

var GAMMA = 0.99;

var BUFFER_SIZE = 2048;
var BATCH_SIZE = 256;
var num_actions = 4;
var NUM_STATE = 8;
var HIDDEN_SIZE = 128;
var NUM_LAYERS = 2;
var ENTROPY_LOSS = 5e-3;
var var LR = 1e-4;  // Lower lr stabilises training greatly

var (DUMMY_ACTION, DUMMY_VALUE) = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1));



} static object exponential_average(object old, object new, object b1){
    return old * b1 + (1 - b1) * new;


} static object proximal_policy_optimization_loss(object advantage, object old_prediction) {
    static object loss(object y_true, object y_pred) {
        prob = np.sum(y_true * y_pred, axis: -1);
        old_prob = np.sum(y_true * old_prediction, axis: -1);
        r = prob / (old_prob + 1e-10);
        return -np.mean(np.minimum(r * advantage, np.clip(r, min_value: 1 - LOSS_CLIPPING, max_value: 1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * np.log(prob + 1e-10)));
    }
    return loss;



} static object proximal_policy_optimization_loss_continuous(object advantage, object old_prediction) {
    static object loss(object y_true, object y_pred) {
        var = np.square(NOISE);
        pi = 3.1415926;
        denom = np.sqrt(2 * pi * var);
        prob_num = np.exp(-np.square(y_true - y_pred) / (2 * var));
        old_prob_num = np.exp(-np.square(y_true - old_prediction) / (2 * var));

        prob = prob_num / denom;
        old_prob = old_prob_num / denom;
        r = prob / (old_prob + 1e-10);

        return -np.mean(np.minimum(r * advantage, np.clip(r, min_value: 1 - LOSS_CLIPPING, max_value: 1 + LOSS_CLIPPING) * advantage));
    }
    return loss;
}


class Agent {
    static object Agent(){
        this.critic = this.build_critic();
        if (CONTINUOUS == false)
            this.actor = this.build_actor();
        else
            this.actor = this.build_actor_continuous();

        this.env = new CartPoleEnv(WinFormEnvViewer.Factory);
        Console.WriteLine(this.env.ActionSpace, "action_space", this.env.ObservationSpace, "observation_space");
        this.episode = 0;
        this.observation = this.env.Reset();
        this.val = false;
        this.reward = new List<double>();
        this.reward_over_time = new List<double>();
        this.name = this.get_name();
        this.writer = SummaryWriter(this.name);
        this.gradient_steps = 0;

    } static object get_name(){
        name = "AllRuns/";
        if (CONTINUOUS == true)
            name += "continous/";
        else
            name += "discrete/";
        name += ENV;
        return name;

    } static object build_actor(){
        state_input = Input(shape: (NUM_STATE,));
        advantage = Input(shape: (1,));
        old_prediction = Input(shape: (NUM_ACTIONS,));

        x = Dense(HIDDEN_SIZE, activation: "tanh").Apply(state_input);
        for (int _ in range(NUM_LAYERS - 1))
            x = Dense(HIDDEN_SIZE, activation: "tanh").Apply(x);

        out_actions = Dense(NUM_ACTIONS, activation: "softmax", name: "output").Apply(x);

        model = Model(inputs:[state_input, advantage, old_prediction], outputs:[out_actions]);
        model.compile(optimizer: Adam(lr: LR), loss:[proximal_policy_optimization_loss(advantage: advantage, old_prediction: old_prediction)]);
        model.summary();

        return model;

    } static object build_actor_continuous(){
        state_input = Input(shape: (NUM_STATE,));
        advantage = Input(shape: (1,));
        old_prediction = Input(shape: (NUM_ACTIONS,));

        x = Dense(HIDDEN_SIZE, activation: "tanh").Apply(state_input);
        for (int _ in range(NUM_LAYERS - 1))
            x = Dense(HIDDEN_SIZE, activation: "tanh").Apply(x);

        out_actions = Dense(NUM_ACTIONS, name: "output", activation: "tanh").Apply(x);

        model = Model(inputs:[state_input, advantage, old_prediction], outputs:[out_actions]);
        model.compile(optimizer: Adam(lr: LR), loss:[proximal_policy_optimization_loss_continuous(advantage: advantage, old_prediction: old_prediction)]);
        model.summary();

        return model;

    } static object build_critic(){

        state_input = Input(shape: (NUM_STATE,));
        x = Dense(HIDDEN_SIZE, activation: "tanh").Apply(state_input);
        for (int _ in range(NUM_LAYERS - 1))
            x = Dense(HIDDEN_SIZE, activation: "tanh").Apply(x);

        out_value = Dense(1).Apply(x);

        model = Model(inputs:[state_input], outputs:[out_value]);
        model.compile(optimizer: Adam(lr: LR), loss: "mse");

        return model;

    } static object reset_env(){
        this.episode += 1;
        if (this.episode % 100 == 0)
            this.val = true;
        else
            this.val = false;
        this.observation = this.env.Reset();
        this.reward = new List<double>();

    } static object get_action(){
        p = this.actor.predict([this.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION]);
        if (this.val == false)
            action = np.random.choice(NUM_ACTIONS, probabilities: np.nan_to_num(p[0]));
        else
            action = np.argmax(p[0]);
        action_matrix = np.zeros(NUM_ACTIONS);
        action_matrix[action] = 1;
        return action, action_matrix, p;

    } static object get_action_continuous(){
        p = this.actor.predict([this.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION]);
        if (this.val == false)
            action = action_matrix = p[0] + np.random.normal(loc: 0, scale: NOISE, size: p[0].shape);
        else
            action = action_matrix = p[0];
        return (action, action_matrix, p);

    } static object transform_reward(){
        if (this.val == true)
            this.writer.add_scalar("Val episode reward", np.array(this.reward).sum(), this.episode);
        else
            this.writer.add_scalar("Episode reward", np.array(this.reward).sum(), this.episode);
        for (int j in range(len(this.reward) - 2, -1, -1))
            this.reward[j] += this.reward[j + 1] * GAMMA;

    } static object get_batch(){
        batch = [[], [], [], []];

        tmp_batch = [[], [], []];
        while (len(batch[0]) < BUFFER_SIZE){
            if (CONTINUOUS==false)
                action, action_matrix, predicted_action = this.get_action();
            else
                action, action_matrix, predicted_action = this.get_action_continuous();
            var (observation, reward, done, info) = this.env.Step(action);
            this.reward.Add(reward);

            tmp_batch[0].Add(this.observation);
            tmp_batch[1].Add(action_matrix);
            tmp_batch[2].Add(predicted_action);
            this.observation = observation;

            if (done)
                this.transform_reward();
                if (this.val==false)
                    for (int i in range(len(tmp_batch[0]))){
                        var (obs, action, pred) = (tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]);
                        r = this.reward[i];
                        batch[0].Add(obs);
                        batch[1].Add(action);
                        batch[2].Add(pred);
                        batch[3].Add(r);
                    }
                }
                tmp_batch = [[], [], []];
                this.reset_env();
            }
        }
        var (obs, action, pred, reward) = (np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1)));
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]));
        return (obs, action, pred, reward);

    } static object run(){
        while (this.episode < EPISODES){
            obs, action, pred, reward = this.get_batch();
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE];
            old_prediction = pred;
            pred_values = this.critic.predict(obs);

            advantage = reward - pred_values;

            actor_loss = this.actor.fit([obs, advantage, old_prediction], [action], batch_size:BATCH_SIZE, shuffle:true, epochs:EPOCHS, verbose:false);
            critic_loss = this.critic.fit([obs], [reward], batch_size:BATCH_SIZE, shuffle:true, epochs:EPOCHS, verbose:false);
            this.writer.add_scalar("Actor loss", actor_loss.history["loss"][-1], this.gradient_steps);
            this.writer.add_scalar("Critic loss", critic_loss.history["loss"][-1], this.gradient_steps);

            this.gradient_steps += 1;
        }
    }
}

Main(){
    ag = Agent();
    ag.run();
}
