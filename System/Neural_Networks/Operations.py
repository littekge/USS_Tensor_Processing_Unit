import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler as sample
import torchax.export as export
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MODEL_PARAMS:
    MODEL: torch.nn.Module = None
    MODEL_NAME: str = None
    RUN_NAME: str = None
    TRAIN: bool = False
    RUN: bool = True
    EXPORT: bool = False

@dataclass
class TRAINING_PARAMS:
    SUBSET_SIZE: int = None # size of the training subset for each epoch
    NUM_EPOCHS: int = 10 # number of epochs to train on
    BATCH_SIZE: int = 4 # number of images processed before calculating loss and updating weights
    LEARNING_RATE: float = 0.001 # learning rate
    MOMENTUM: float = 0.9 # momentum
    LOSS_FUNCTION: str = "MSE" # which loss function to use in training
    CLASSIFICATION_MODE: str = "regression" # type of correctness classification to use in testing

def Start(model_params, training_params, trainSet=None, testSet=None, exportSet=None):
    # Set up warnings
    import warnings
    warnings.filterwarnings(
        action='ignore',
        category=DeprecationWarning,
        module=r'.*'
    )
    warnings.filterwarnings(
        action='ignore',
        category=FutureWarning,
        module=r'.*'
    )
    

    # assertions
    assert isinstance(model_params, MODEL_PARAMS) # asserts that model_params is correct type
    assert model_params.MODEL != None
    assert model_params.MODEL_NAME != None
    assert model_params.RUN_NAME != None
    if model_params.TRAIN: assert trainSet != None # assert that trainSet exists if training
    if model_params.RUN: assert testSet != None # assert that testSet exisits if running
    if model_params.EXPORT: assert exportSet != None # assert that exportSet exisits if exporting
    if model_params.EXPORT: assert testSet != None # export calibration reuses the test set to measure activation ranges for M

    # determining path variables
    CURRENT_DIR = Path(__file__).parent # determining file path
    PATH = CURRENT_DIR / model_params.MODEL_NAME / str(model_params.MODEL_NAME + "_" + model_params.RUN_NAME) # setting save path

    if model_params.TRAIN: Train(model=model_params.MODEL, dataset=trainSet, save_file_path=PATH, params=training_params)
    if model_params.EXPORT: Save(model=model_params.MODEL, sample_input=exportSet, save_file_path=PATH, calibration_set=testSet)
    if model_params.RUN: Run(model=model_params.MODEL, dataset=testSet, load_file_path=PATH, params=training_params)
    
    

def Train(model, dataset, save_file_path, params):
    # assertions
    assert isinstance(params, TRAINING_PARAMS) # assert that params is correct type
    assert params.SUBSET_SIZE != None
    assert 0 < params.SUBSET_SIZE <= len(dataset) # subset size must be less than or equal to dataset size
    assert 0 < params.BATCH_SIZE <= params.SUBSET_SIZE # batch size must be less than or equal to subset size
    # constants def
    PRINT_INTERVAL = max(1, int(params.SUBSET_SIZE/params.BATCH_SIZE/10)) # how often training data should be printed

    # different loss functions
    match params.LOSS_FUNCTION:
        case "MSE": criterion = nn.MSELoss()
        case "CrossEntropy": criterion = nn.CrossEntropyLoss() 
        case _: 
            print("invalid loss_type, defaulting to MSELoss...")
            criterion = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), params.LEARNING_RATE, params.MOMENTUM) # learning driver to adjust weights

    for epoch in range(params.NUM_EPOCHS):
        running_loss = 0.0 # loss accumulated during epoch
        indices = torch.randperm(len(dataset))[:params.SUBSET_SIZE].tolist() # generate a 
        sampler = sample(indices)
        trainLoader = torch.utils.data.DataLoader(dataset, batch_size=params.BATCH_SIZE, sampler=sampler, num_workers=2)

        for i, data in enumerate(trainLoader, 0): # loop through training data
            inputs, labels = data # get input data and labels
            optimizer.zero_grad() # zeroing gradient in between iterations
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # calculate loss
            loss.backward() # backwards propogation to calculate gradients
            optimizer.step() # adjust weights

            # print statistics
            running_loss += loss.item()
            if i % PRINT_INTERVAL == PRINT_INTERVAL-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / PRINT_INTERVAL))
                running_loss = 0.0

    print('Finished Training')

    # saving data
    torch.save(model.state_dict(), Path(str(save_file_path) + ".pth"))
    print("Model saved successfully!")
    


# runs the neural network with relevant parameters
def Run(model, dataset, load_file_path, params):
    load_file_path = Path(str(load_file_path) + ".pth")
    # assertions
    assert isinstance(params, TRAINING_PARAMS) # assert that params is correct type
    assert load_file_path.is_file() # assert that the weights file exists
    assert 0 < params.BATCH_SIZE < len(dataset) # Batch size cannot be larger than dataset

    # constants def
    PRINT_INTERVAL = max(1, int(len(dataset) / params.BATCH_SIZE / 10)) # determining print interval

    model.load_state_dict(torch.load(load_file_path, weights_only=True)) # loading model
    testLoader = torch.utils.data.DataLoader(dataset, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=2) # loading test dataset
    # calculating and printing the accuracy of the model
    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(testLoader, 0):
            inputs, labels = data
            outputs = model(inputs)

            match params.CLASSIFICATION_MODE:
                case "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                case "regression":
                    tolerance = 0.1
                    correct += (torch.abs(outputs - labels) < tolerance).sum().item()
                    total += labels.numel()

                case _:
                    print("invalid classification mode, defaulting to regression...")
                    tolerance = 0.1
                    correct += (torch.abs(outputs - labels) < tolerance).sum().item()
                    total += labels.numel()

            if i % PRINT_INTERVAL == PRINT_INTERVAL - 1:
                print("[%d] correct: %d total: %d" % (i + 1, correct, total))

    print('\nAccuracy of the network: %d %%' % (
    100 * correct / total))
   
def _calibrate_scales(model, dataset, batch_size=64):
    """Measure the per-layer requantization multiplier M for every Linear/Conv layer.

    For each layer M = (S_in * S_w) / S_out, where S = r_max / 127 is the symmetric
    int8 scale of that tensor. S_w comes from the weights; S_in / S_out come from
    activation ranges measured over `dataset` (the calibration set).

    S_in is CHAINED, not measured per layer: S_in(n) = S_out(n-1). On the TPU the
    int8 tensor handed between layers carries the previous layer's output scale, and
    ReLU/pooling are scale-preserving integer ops that never requantize -- so the
    scale the hardware actually sees at a layer's input is the prior layer's S_out.
    The first layer's S_in comes from the network input range.

    Returns {"<layer>.weight": {"M", "S_in", "S_w", "S_out"}}, keyed to match the
    weight tensor names stored in the .npz.
    """
    layers = [(name, mod) for name, mod in model.named_modules()
              if isinstance(mod, (nn.Linear, nn.Conv2d))]
    assert layers, "No Linear/Conv2d layers found to calibrate."
    layer_by_name = dict(layers)

    absmax_in = {name: 0.0 for name, _ in layers}
    absmax_out = {name: 0.0 for name, _ in layers}
    fire_order = []  # execution order, recorded from the order hooks first fire

    def make_hook(nm):
        def hook(_module, inp, out):
            if nm not in fire_order:
                fire_order.append(nm)
            absmax_in[nm] = max(absmax_in[nm], inp[0].detach().abs().max().item())
            absmax_out[nm] = max(absmax_out[nm], out.detach().abs().max().item())
        return hook

    handles = [mod.register_forward_hook(make_hook(name)) for name, mod in layers]
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(inputs)
    for h in handles:
        h.remove()

    scales = {}
    prev_S_out = None
    for idx, name in enumerate(fire_order):
        S_w = layer_by_name[name].weight.detach().abs().max().item() / 127.0
        S_out = absmax_out[name] / 127.0
        S_in = (absmax_in[name] / 127.0) if idx == 0 else prev_S_out
        # A dead layer (all-zero activations over the calibration set) has no
        # meaningful output scale; emit M=0 rather than divide by zero so the
        # downstream tool can surface it.
        M = 0.0 if S_out == 0.0 else (S_in * S_w) / S_out
        scales[f"{name}.weight"] = {"M": M, "S_in": S_in, "S_w": S_w, "S_out": S_out}
        prev_S_out = S_out
    return scales


def Save(model, sample_input, save_file_path, calibration_set):
    # Export the TRAINED network. Load weights from the .pth exactly as Run does so
    # that `-e` exports an already-trained model (and matches the just-trained weights
    # when `-t` runs in the same invocation), rather than exporting random init.
    load_path = Path(str(save_file_path) + ".pth")
    assert load_path.is_file(), (
        f"Cannot export: no trained weights found at {load_path}. "
        f"Train the model first with -t, or ensure the .pth exists."
    )
    model.load_state_dict(torch.load(load_path, weights_only=True))
    model = model.eval()
    exportedModel = torch.export.export(model, sample_input) # converted to ExportedProgram
    torch.export.save(exportedModel, Path(str(save_file_path) + ".pt2")) # save as pt2 archive
    print("Model exported to .pt2 successfully!")
    weights, stablehlo = export.exported_program_to_stablehlo(exportedModel)
    with open(Path(str(save_file_path) + ".mlir"), "w") as out:
        out.write(stablehlo.mlir_module())
    print("Model exported to .mlir successfully!")

    # Dump the raw weight values alongside the .mlir/.pt2.
    #
    # WHY: the .mlir module only stores tensor shapes/types (the @main signature) -- it does
    # NOT contain the actual weight numbers. Those live only in the exported program's state.
    # We persist them here so the downstream Synth_Flow step can quantize them to int8 and
    # build the hardware load packets without having to re-export the model.
    #
    # export_raw=True returns matched (names, states, func): the names come from the
    # graph signature (parameters + buffers) and line up positionally with the leading
    # @main arguments %arg0..%argN-2 (the final @main argument is the runtime input, not a
    # weight). states are torch tensors here (the conversion to jax arrays happens only in
    # the non-raw path), so we can move them straight to numpy.
    names, states, _ = export.exported_program_to_jax(exportedModel, export_raw=True)
    weight_arrays = {name: tensor.detach().cpu().numpy() for name, tensor in zip(names, states)}
    # Save an explicit ordered name list so the downstream step can preserve @main-arg order
    # for sequential address assignment (dict order is preserved, but this is unambiguous).
    weight_arrays["__order__"] = np.array(list(names))

    # v0.3 requantization scales: measure per-layer M = (S_in*S_w)/S_out over the
    # calibration set and store it ALONGSIDE the weights. These extra keys are
    # ignored by the weight loader (Process_Weights reads only __order__ and the
    # named weight tensors), so they cannot disrupt the existing weight export.
    # Keys are prefixed and suffixed with the layer's weight name so the assembler
    # can attach each M to the matmul that consumes that weight. The component
    # scales are kept for traceability/debugging.
    scales = _calibrate_scales(model, calibration_set)
    for weight_name, s in scales.items():
        weight_arrays[f"__M__{weight_name}"] = np.float64(s["M"])
        weight_arrays[f"__scales__{weight_name}"] = np.array(
            [s["S_in"], s["S_w"], s["S_out"]], dtype=np.float64
        )

    np.savez(Path(str(save_file_path) + ".weights.npz"), **weight_arrays)
    print(weight_arrays)
    print("Weights exported to .weights.npz successfully!")

    