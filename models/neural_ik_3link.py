import torch
from torch import nn


def forward_kinematics_3link_torch(theta):
    theta1 = theta[:, 0]
    theta2 = theta[:, 1]
    theta3 = theta[:, 2]

    end_effector_x = (
        torch.cos(theta1)
        + torch.cos(theta1 + theta2)
        + torch.cos(theta1 + theta2 + theta3)
    )
    end_effector_y = (
        torch.sin(theta1)
        + torch.sin(theta1 + theta2)
        + torch.sin(theta1 + theta2 + theta3)
    )

    return torch.stack([end_effector_x, end_effector_y], dim=1)


#defines a neural network class, inherits from PyTorch nn base class
class NeuralIK3Link(nn.Module):
    def __init__(self): #defines constructor for class
        super().__init__() #calls the constructor of nn.Module

        self.network = nn.Sequential(
            #Creates sequence of layers
            nn.Linear(8, 256),
            nn.ReLU(), #H1
            nn.Linear(256, 256),
            nn.ReLU(), #H2
            nn.Linear(256, 256),
            nn.ReLU(), #H3
            nn.Linear(256, 3), #Y_hat
        ) #Full network is now stored in self.network

    #Runs input through the entire network.
    #Returns predicted delta theta updates.
    def forward(self, x):
        single_input = x.ndim == 1
        if single_input:
            x = x.unsqueeze(0)

        target = x[:, 0:2]
        theta = x[:, 2:5]

        current_position = forward_kinematics_3link_torch(theta)

        #Cartesian error is what direction and distance the arm needs to move
        #Dividing by 6 keeps its possible range roughly between -1 and 1
        cartesian_error = (target - current_position) / 6.0

        #sin and cos avoid the sudden jump between angles -pi and pi
        angle_features = torch.stack(
            [torch.sin(theta), torch.cos(theta)],
            dim=2,
        ).flatten(start_dim=1)

        features = torch.cat([cartesian_error, angle_features], dim=1)
        predicted_delta_theta = self.network(features)

        if single_input:
            return predicted_delta_theta[0]

        return predicted_delta_theta
