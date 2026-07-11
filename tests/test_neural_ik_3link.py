import numpy as np
import torch
from torch import nn

from models.neural_ik_3link import NeuralIK3Link
from solvers.neural_ik_3link import neural_solver_3link


def test_neural_ik_3link_output_shape():
    model = NeuralIK3Link()
    model_input = torch.zeros((4, 5))

    output = model(model_input)

    assert output.shape == (4, 3)


def test_neural_solver_3link_applies_predicted_update():
    #This simple model always predicts [pi/2, 0, 0]
    model = nn.Linear(5, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([np.pi / 2, 0.0, 0.0])

    #Rotating theta1 by pi/2 moves the straight arm from [3, 0] to [0, 3]
    result = neural_solver_3link(
        model,
        target=[0.0, 3.0],
        initial_theta=[0.0, 0.0, 0.0],
        tolerance=1e-5,
        max_iterations=5,
        max_step_norm=None,
    )

    assert result["success"]
    assert result["iterations"] == 1
    assert result["final_error"] < 1e-5
    assert np.allclose(result["theta"], [np.pi / 2, 0.0, 0.0])
