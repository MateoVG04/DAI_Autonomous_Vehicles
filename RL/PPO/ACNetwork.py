import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_steering_dim, output_steering_dim, input_speed_dim, output_speed_dim):
        super().__init__()
        # The neural network gets images and info from the computer vision
        # CNN_branch: This branch is for image processing
        """
        The CNN will output steering
        """
        self.cnn1 = nn.Conv2d(input_steering_dim, 256, kernel_size=3, stride=1, padding=1)
        self.cnn2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.cnn3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))  # robust to input HxW
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_steering_dim)
        self.dropout = nn.Dropout(0.1)

        # Distance branch for Adaptive Cruise Control (ACC)
        """
        This will take the output of the CNN + the distance to vehicle to calculate the speed 
        + speed signs and later pedestrians and red lights can be added to change the driving speed. 
        """
        self.speed_fc1 = nn.LazyLinear(64)
        self.speed_fc2 = nn.Linear(64, 128)
        self.speed_fc3 = nn.Linear(128, 128)
        self.speed_fc4 = nn.Linear(128, 64)
        self.speed_fc5 = nn.Linear(64, output_speed_dim)

        action_dim = output_steering_dim + output_speed_dim
        #Make the shared features always the same length
        self.head_reduce = nn.Linear(action_dim, 128)
        # Actor head
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(128, 1)

    def cnn_foward(self, images:torch.Tensor) -> tuple:
        steering = self.pool(F.relu(self.cnn1(images)))
        steering = self.pool(F.relu(self.cnn2(steering)))
        steering = self.pool(F.relu(self.cnn3(steering)))
        steering = self.adapt(steering)
        # steering = steering.view((-1, 64 * 4 * 4))
        steering = steering.view(steering.size(0), -1)
        steering = F.relu(self.fc1(steering))
        steering = self.dropout(steering)
        steering_feat = F.relu(self.fc2(steering))
        # No activation, no bounds -> let the agent decide the bounds themselves
        steer_out = self.fc3(steering_feat)
        return steering_feat, steer_out

    @staticmethod
    def ensure_2d(x):
        if x is None:
            return None
        if x.dim() == 1:  # (B,) -> (B,1)
            x = x.unsqueeze(1)
        return x

        # to accommodate the different branches for different applications
    def forward(self, images:torch.Tensor, current_speed, distances_to_vehicle, speed_signs, pedestrians, red_lights):
        """
        Forward the data through the different neural networks, fuse them and calculate the critic score on
        the steering and speed regulation.
        :param images: can be an image of the road, or a preprocessed image with computer vision of the lanes
        :param distances_to_vehicle:
        :param speed_signs:
        :param pedestrians:
        :param red_lights:
        :return actor_mean:
        :return actor_std:
        :return state_value:
        """
        steer_feat, steer_out = self.cnn_foward(images)

        data = [
            self.ensure_2d(steer_feat),
            self.ensure_2d(current_speed),
            self.ensure_2d(distances_to_vehicle),
            self.ensure_2d(speed_signs),
            self.ensure_2d(pedestrians),
            self.ensure_2d(red_lights)
        ]
        data = [d for d in data if d is not None]

        # Fusing the data for the speed network
        fused = torch.cat(data, 1)

        speed = F.relu(self.speed_fc1(fused))
        speed = F.relu(self.speed_fc2(speed))
        speed = F.relu(self.speed_fc3(speed))
        speed = F.relu(self.speed_fc4(speed))
        # No ReLU because we might also want negative throttle = braking
        speed_out = self.speed_fc5(speed)

        shared_features = torch.cat([steer_out, speed_out],dim=1)
        shared_features = self.head_reduce(shared_features)

        mean = self.actor_mean(shared_features)
        log_std = self.actor_log_std.clamp(-20,2)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean)
        value = self.critic(shared_features)
        return mean, std, value

    @torch.no_grad()
    def sample_action(self, images, current_speed,distances_to_vehicle=None, speed_signs=None, pedestrians=None, red_lights=None):
        mean, std, _ = self(images, current_speed, distances_to_vehicle, speed_signs, pedestrians, red_lights)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        #adjustment for tanh squashing
        log_prob = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob
