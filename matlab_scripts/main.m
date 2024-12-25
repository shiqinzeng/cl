% Add the Python script directory to MATLAB's Python path - change to your
% own path
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
%group1 = torch.randn(py.tuple([int32(1), int32(7)]), pyargs('dtype', torch.float32)).to(device);  % Group 1 input
%group2 = torch.randn(py.tuple([int32(1), int32(7)]), pyargs('dtype', torch.float32)).to(device);  % Group 2 input
%discrete = torch.randn(py.tuple([int32(1), int32(6)]), pyargs('dtype', torch.float32)).to(device); % Discrete input
% Specify the file path
filePath = '500kHzNC_train.xlsx';  % Replace with your Excel file path

% Read the Excel file into a table
dataTable = readtable(filePath);

% Identify the headers
headers = dataTable.Properties.VariableNames;

% Filter columns based on headers
group1Idx = startsWith(headers, 'H');  % Columns starting with 'H'
group2Idx = startsWith(headers, 'U');  % Columns starting with 'U'
excludeIdx = strcmp(headers, 'Animalid') | strcmp(headers, 'BB_500kHz');  % Exclude specific columns
discreteIdx = ~(group1Idx | group2Idx | excludeIdx);  % All other columns except excluded

% Extract the data for each group
group1Data = table2array(dataTable(:, group1Idx));  % Group1: Columns starting with 'H'
group2Data = table2array(dataTable(:, group2Idx));  % Group2: Columns starting with 'U'
discreteData = table2array(dataTable(:, discreteIdx));  % Remaining columns except 'Animalid' and 'BB_500kHz'

batch_size = 1;
num_batches = ceil(size(group1Data, 1) / batch_size);
% Define stats for normalization
group1_mean = torch.tensor([4.9914, 4.9280, 4.8687, 4.9533, 4.9859, 4.8852, 4.8068], pyargs('dtype', torch.float32));
group1_var = torch.tensor([0.2617, 0.2651, 0.2585, 0.2740, 0.2683, 0.2582, 0.2302], pyargs('dtype', torch.float32));
group2_mean = torch.tensor([3.5835, 3.5487, 3.5594, 3.5823, 3.6443, 3.6465, 3.6531], pyargs('dtype', torch.float32));
group2_var = torch.tensor([0.0840, 0.0912, 0.0931, 0.1064, 0.1243, 0.1216, 0.1177], pyargs('dtype', torch.float32));
discrete_mean = torch.tensor([0.1345, -2.6016, 0.5350, 41.7834, 65.7800, 0.6913], pyargs('dtype', torch.float32));
discrete_var = torch.tensor([3.4649, 2.0950, 0.24878, 220.07, 1442.3, 0.048473], pyargs('dtype', torch.float32));
epsilon = torch.tensor(1e-6, pyargs('dtype', torch.float32));

% Initialize predictions
predictions = [];

for batch_idx = 1:num_batches
    % Extract the batch
    start_idx = (batch_idx - 1) * batch_size + 1;
    end_idx = min(batch_idx * batch_size, size(group1Data, 1));
    group11 = group1Data(start_idx:end_idx, :);
    group21 = group2Data(start_idx:end_idx, :);
    discrete1 = discreteData(start_idx:end_idx, :);
    
    % Convert batch to PyTorch tensors
    group1_tensor = torch.tensor(group11, pyargs('dtype', torch.float32)).to(device);
    group2_tensor = torch.tensor(group21, pyargs('dtype', torch.float32)).to(device);
    discrete_tensor = torch.tensor(discrete1, pyargs('dtype', torch.float32)).to(device);
    
    % Normalize tensors using PyTorch operations
    group1_tensor = torch.div(group1_tensor - group1_mean, torch.sqrt(group1_var + epsilon));
    group2_tensor = torch.div(group2_tensor - group2_mean, torch.sqrt(group2_var + epsilon));
    discrete_tensor = torch.div(discrete_tensor - discrete_mean, torch.sqrt(discrete_var + epsilon));
    
    % Add batch dimension to tensors
    group1_tensor = group1_tensor.unsqueeze(int32(0));  % Transform to 2D tensor
    group2_tensor = group2_tensor.unsqueeze(int32(0));
    discrete_tensor = discrete_tensor.unsqueeze(int32(0));
    
    % Perform inference using helper function
    output = helper.infer_with_no_grad(model, group1_tensor, group2_tensor, discrete_tensor);
    
    % Convert PyTorch tensor to MATLAB array
    output_array = double(output.numpy());
    
    % Get the predicted class
    [~, prediction] = max(output_array, [], 2);
    
    % Display the predicted class
    disp(['Predicted class: ', num2str(prediction)]);
    predictions = [predictions; prediction];
end
