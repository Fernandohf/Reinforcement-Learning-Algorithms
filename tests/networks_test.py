from bistrain.networks.actors import FCActorContinuous, FCActorDiscrete
from bistrain.networks.critics import FCCritic  # LSTMCritics
import torch


class TestFCActorContinuous():
    """
    Testing Actor with fully connected layers in continuous actions.
    """
    def test_init1(self):
        a = FCActorContinuous(4, 2)
        assert len(a.layers) == 4

    def test_init2(self):
        a = FCActorContinuous(4, 2, (4, 16, 4))
        assert len(a.layers) == 5

    def test_init3(self):
        a = FCActorContinuous(4, 2, hidden_activation='sigmoid')
        assert a.hidden_activation == torch.sigmoid

    def test_output1(self):
        a = FCActorContinuous(4, 2)
        s = torch.Tensor([[1, 2, 3, 4]])
        out, _ = a(s)
        assert out.shape == (1, 2)

    def test_output2(self):
        a = FCActorContinuous(2, 16)
        s = torch.Tensor([[1, 2], [1, 2], [3, 4]])
        out, _ = a(s)
        assert out.shape == (3, 16)

    def test_output3(self):
        a = FCActorContinuous(4, 2, output_loc_scaler=1)
        s = torch.Tensor([[1, 2, 3, 4]])
        out, _ = a(s)
        assert out.abs().max() <= 1

    def test_output4(self):
        a = FCActorContinuous(4, 2,
                              output_loc_activation='sigmoid',
                              output_loc_scaler=1,
                              output_range=(0, 1))
        s = torch.Tensor([[1, 2, 3, 4]])
        out, _ = a(s)
        assert (all(out.view(-1) <= 1) and all(out.view(-1) >= 0))

    def test_output5(self):
        a = FCActorContinuous(4, 2,
                              output_loc_activation='tanh',
                              output_loc_scaler=5,
                              output_range=(-5, 5))
        s = torch.Tensor([[1, 2, 3, 4]])
        out, _ = a(s)
        assert (all(out.view(-1) <= 5) and all(out.view(-1) >= -5))


class TestFCActorDiscrete():
    """
    Testing Actor with fully connected layers in discrete actions.
    """
    def test_init1(self):
        a = FCActorDiscrete(4, 2)
        assert len(a.layers) == 3

    def test_init2(self):
        a = FCActorDiscrete(4, 2, (4, 16, 4))
        assert len(a.layers) == 4

    def test_init3(self):
        a = FCActorDiscrete(4, 2, hidden_activation='sigmoid')
        assert a.hidden_activation == torch.sigmoid

    def test_output1(self):
        a = FCActorDiscrete(4, 2)
        s = torch.Tensor([[1, 2, 3, 4]])
        out, _ = a(s)
        assert out.shape == (1, 1)

    def test_output2(self):
        a = FCActorDiscrete(2, 1)
        s = torch.Tensor([[1, 2], [1, 2], [3, 4]])
        out, _ = a(s)
        assert out.shape == (3, 1)

    def test_output3(self):
        a = FCActorDiscrete(4, 5)
        s = torch.Tensor([[1, 2, 3, 4]])
        out, _ = a(s)
        assert all(out.view(-1) <= 4) and all(out.view(-1) >= 0)

    def test_output4(self):
        a = FCActorDiscrete(4, 2)
        s = torch.Tensor([[1, 2, 3, 4]])
        out, _ = a(s)
        assert all(out.view(-1) <= 1) and all(out.view(-1) >= 0)


class TestFCCritic():
    """
    Testing Actor with fully connected layers in discrete actions.
    """
    def test_init1(self):
        c = FCCritic(4, 2)
        assert len(c.layers) == 3

    def test_init2(self):
        c = FCCritic(4, 2, (4, 16, 4))
        assert len(c.layers) == 4

    def test_init3(self):
        c = FCCritic(4, 2, hidden_activation='sigmoid')
        assert c.hidden_activation == torch.sigmoid

    def test_output1(self):
        c = FCCritic(4, 2)
        s = torch.Tensor([[1, 2, 3, 4]])
        a = torch.Tensor([[3, 4]])
        out = c(s, a)
        assert out.shape == (1, 1)

    def test_output2(self):
        c = FCCritic(2, 1)
        s = torch.Tensor([[1, 2], [1, 2], [3, 4]])
        a = torch.Tensor([[3], [2], [4]])
        out = c(s, a)
        assert out.shape == (3, 1)
