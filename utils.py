import torch
import torch.nn as nn

def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    if len(memory) < batch_size:
        return None
    transitions = memory.sample(batch_size)

    state_batch = torch.stack([torch.tensor(t.state) for t in transitions]).float().to(device)
    action_batch = torch.tensor([t.action for t in transitions], dtype=torch.int64).unsqueeze(1).to(device)
    reward_batch = torch.tensor([t.reward for t in transitions], dtype=torch.float32).to(device)
    next_state_batch = torch.stack([torch.tensor(t.next_state) for t in transitions]).float().to(device)
    done_batch = torch.tensor([t.done for t in transitions], dtype=torch.float32).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch)
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0]
    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
