% Add the Python script directory to MATLAB's Python path
if count(py.sys.path, "C:\Users\Shiqin\Downloads\cl") == 0
    insert(py.sys.path, int32(0), "C:\Users\Shiqin\Downloads\cl");
end

% Import the Python module
network = py.importlib.import_module('network');

% Import PyTorch
torch = py.importlib.import_module('torch');
helper = py.importlib.import_module('helper');
py.importlib.reload(helper);  % Reload the helper module to ensure updates are recognized

model = network.BoostedMLP( ...
    network.EnhancedGroupedMLP, ...  % base_model
    py.list([int32(7), int32(7)]), ... % group_dims as integers
    int32(6), ...                    % discrete_dim
    int32(20), ...                   % combined_dim
    int32(10), ...                   % num_boosting_steps
    0.3, ...                         % dropout_prob
    int32(4));                       % num_heads

% Load the model's state dictionary
state_dict = torch.load('best_false_negatives_model_epoch_75.pth', pyargs('map_location', 'cpu'));
model.load_state_dict(state_dict);
model.eval();

% Transfer the model to the appropriate device (CPU in this case)
device = torch.device('cpu');
model.to(device);

% Transfer the model to the appropriate device
device = torch.device('cpu');
model.to(device);

% Create float32 input tensors
group1 = torch.randn(py.tuple([int32(1), int32(7)]), pyargs('dtype', torch.float32)).to(device);  % Group 1 input
group2 = torch.randn(py.tuple([int32(1), int32(7)]), pyargs('dtype', torch.float32)).to(device);  % Group 2 input
discrete = torch.randn(py.tuple([int32(1), int32(6)]), pyargs('dtype', torch.float32)).to(device); % Discrete input


% Perform inference using helper function
output = helper.infer_with_no_grad(model, group1, group2, discrete);

% Convert PyTorch tensor to MATLAB array
output_array = double(output.numpy());

% Get the predicted class
[~, prediction] = max(output_array, [], 2);

% Display the predicted class
disp(['Predicted class: ', num2str(prediction)]);