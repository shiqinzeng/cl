import torch
from network import EnhancedGroupedMLP, BoostedMLP
# Load the trained model
boosted_model = BoostedMLP(
    base_model=EnhancedGroupedMLP,
    group_dims=[7, 7],
    discrete_dim=6,
    combined_dim=20,
    num_boosting_steps =10,
    dropout_prob=0.3,
    num_heads = 4)
boosted_model.load_state_dict(torch.load("best_false_negatives_model_epoch_75.pth",map_location=torch.device('cpu')))
boosted_model.eval()

# # Example inference
# with torch.no_grad():
#     test_group1 = torch.randn(1, 7) # Single sample, 10 features
#     test_group2 = torch.randn(1, 7)   # Single sample, 8 features
#     test_discrete = torch.randn(1, 6) # Single sample, 5 features

#     output = boosted_model(test_group1, test_group2, test_discrete)
#     prediction = torch.argmax(output, dim=1).item()
#     print(f"Predicted class: {prediction}")


def infer(group1, group2, discrete):
    """
    Perform inference using the trained PyTorch model.

    Args:
        group1 (list or array): Features for group 1.
        group2 (list or array): Features for group 2.
        discrete (list or array): Discrete features.

    Returns:
        int: Predicted class.
    """
    # Convert inputs to tensors
    group1 = torch.tensor(group1).float().unsqueeze(0)  # Add batch dimension
    group2 = torch.tensor(group2).float().unsqueeze(0)
    discrete = torch.tensor(discrete).float().unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = boosted_model([group1, group2], discrete)
        prediction = torch.argmax(output, dim=1).item()

    return prediction
