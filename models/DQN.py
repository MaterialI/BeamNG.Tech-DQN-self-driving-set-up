import torch

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, feature_dim=20, num_actions=9):
        super(DQN, self).__init__()
        self.lstm = nn.LSTM(input_size=21, hidden_size=256, batch_first=True)
        self.lstm_fc = nn.Linear(256, 256)

        # First image branch
        # Add a simple ResidualBlock definition first
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

            def forward(self, x):
                identity = x
                out = self.relu(self.conv1(x))
                out = self.conv2(out)
                return self.relu(out + identity)

        # Updated branches with pooling and residuals
        self.img_branch1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1),  # 300x300x3 -> 296x296x24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 296x296x24 -> 148x148x24
            ResidualBlock(8),  # 148x148x24 -> 148x148x24
            nn.Conv2d(8, 16, kernel_size=5, stride=1),  # 148x148x24 -> 144x144x48
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 144x144x48 -> 72x72x48
            ResidualBlock(16),  # 72x72x48 -> 72x72x48
            nn.Conv2d(16, 16, kernel_size=3, stride=2),  # 72x72x48 -> 35x35x96
            nn.ReLU(),
            ResidualBlock(16),  # 35x35x96 -> 35x35x96
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # 35x35x64 -> 17x17x128
            nn.Flatten(),  # 17x17x128 -> 17*17*128s = 36992
        )

        self.img_branch2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1),  # 300x300x3 -> 296x296x24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 296x296x24 -> 148x148x24
            ResidualBlock(8),  # 148x148x24 -> 148x148x24
            nn.Conv2d(8, 16, kernel_size=5, stride=1),  # 148x148x24 -> 144x144x48
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 144x144x48 -> 72x72x48
            ResidualBlock(16),  # 72x72x48 -> 72x72x48
            nn.Conv2d(16, 16, kernel_size=3, stride=2),  # 72x72x48 -> 35x35x96
            nn.ReLU(),
            ResidualBlock(16),  # 35x35x96 -> 35x35x96
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # 35x35x64 -> 17x17x128
            nn.Flatten(),  # 17x17x128 -> 17*17*128s = 36992
        )
        # Fully connected for non-image features
        self.fc_features = nn.Linear(feature_dim, 16)
        # Combine branches
        self.fc_combined = nn.Sequential(  # Process high level features with dropout to reduce overfitting.
            nn.Linear(17 * 17 * 32 * 2 + 20 * 256, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_actions, bias=True),
        )

    def forward(self, img1, img2, features):
        # Convert images from NHWC to NCHW format
        """
        Input: img1, img2, features
        img1: torch.Tensor of shape (batch_size, 300, 300, 4)
        img2: torch.Tensor of shape (batch_size, 300, 300, 4)

        features: torch.Tensor of shape (batch_size, seq_len, feature_dim)
        Output: q_values
        q_values: torch.Tensor of shape (batch_size, num_actions)"""
        # print("DIMS inference", img1.shape, img2.shape, features.shape)
        img1 = img1.permute(0, 3, 1, 2)
        img2 = img2.permute(0, 3, 1, 2)

        x1 = self.img_branch1(img1)
        x2 = self.img_branch2(img2)

        # Process features: assume shape (batch, seq_len, feature_dim)
        lstm_out, _ = self.lstm(features)
        x3 = F.relu(self.lstm_fc(lstm_out))
        # print()
        x3 = x3.reshape(x3.size(0), -1)

        # print(x1.shape, x2.shape, x3.shape)
        # Concatenate
        x = torch.cat([x1, x2, x3], dim=1)
        q_values = self.fc_combined(x)
        return q_values

    # train based on replay buffer
    def train_step(self, optimizer, loss_fn, batch):
        # Unpack batch
        global data_buffer
        # Separate each component from the batch arrays
        obs_batches = [
            b[0] for b in batch
        ]  # gives index of prev observation inside of the data queue
        action_batches = [b[1] for b in batch]
        reward_batches = [b[2] for b in batch]
        next_obs_batches = [
            b[3] for b in batch
        ]  # gives index of next observation inside of the data queue
        done_batches = [b[4] for b in batch]

        # Combine each observation component into tensors
        img2_batch = torch.cat(
            [(obs["img2"] / 255.0).to(torch.bfloat16) for obs in obs_batches], dim=0
        )
        img1_batch = torch.cat(
            [(obs["img1"] / 255.0).to(torch.bfloat16) for obs in obs_batches], dim=0
        )
        feature_batch = torch.cat([obs["obs"] for obs in obs_batches], dim=0)
        # print("Backrpop dims", img1_batch.shape, img2_batch.shape, feature_batch.shape)
        # Compute current Q-values
        q_values_all = self(img1_batch, img2_batch, feature_batch)
        actions = torch.tensor(action_batches, dtype=torch.long).to("cuda")

        q_values = q_values_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Prepare next observations
        next_img1_batch = torch.cat(
            [(obs["img1"] / 255.0).to(torch.bfloat16) for obs in next_obs_batches],
            dim=0,
        )
        next_img2_batch = torch.cat(
            [(obs["img2"] / 255.0).to(torch.bfloat16) for obs in next_obs_batches],
            dim=0,
        )
        next_feature_batch = torch.cat([obs["obs"] for obs in next_obs_batches], dim=0)
        # print(
        #     "Backrpop dims_next",
        #     next_img1_batch.shape,
        #     next_img2_batch.shape,
        #     next_feature_batch.shape,
        # )
        # Compute next Q-values
        next_q_values_all = self(next_img1_batch, next_img2_batch, next_feature_batch)
        next_q_values = next_q_values_all.max(dim=1).values

        # Targets
        rewards = torch.tensor(reward_batches, dtype=torch.bfloat16).to("cuda")
        dones = torch.tensor(done_batches, dtype=torch.bfloat16).to("cuda")
        target_q_values = rewards + 0.9999 * next_q_values * (1 - dones)

        # Loss and backprop
        loss = loss_fn(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # remove all tensors from cuda
        del (
            img1_batch,
            img2_batch,
            feature_batch,
            next_img1_batch,
            next_img2_batch,
            next_feature_batch,
        )

        return loss.item()
