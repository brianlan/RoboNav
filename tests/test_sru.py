import torch

from robonav.aqua.model.sru import LSTMSRUCell, LSTM_SRU, SRU


def test_cell_gate_equation():
    torch.manual_seed(0)
    cell = LSTMSRUCell(3, 4)
    x = torch.randn(2, 3)
    h = torch.randn(2, 4)
    c = torch.randn(2, 4)

    h_next, c_next = cell(x, h, c)

    # Independent oracle recomputing the gate equations from the raw parameters.
    gates = torch.nn.functional.linear(torch.cat([x, h], dim=1), cell.linear_all.weight, cell.linear_all.bias)
    tx = torch.nn.functional.linear(x, cell.transform_gate.weight, cell.transform_gate.bias)
    i, f, o, g = torch.split(gates, 4, dim=1)
    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
    g_t = torch.tanh(tx * g)
    f = i * (1.0 - (1.0 - f) ** 2) + (1.0 - i) * f**2
    c_ref = f * c + (1.0 - f) * g_t
    h_ref = o * torch.tanh(c_ref)

    torch.testing.assert_close(h_next, h_ref)
    torch.testing.assert_close(c_next, c_ref)


def test_recurrent_module():
    torch.manual_seed(1)
    assert SRU is LSTM_SRU

    bf = LSTM_SRU(input_size=3, hidden_size=5, num_layers=2, batch_first=True)
    tf = LSTM_SRU(input_size=3, hidden_size=5, num_layers=2)
    tf.load_state_dict(bf.state_dict())  # exercises shared state_dict layout
    assert set(tf.state_dict()) >= {
        "cells.0.linear_all.weight",
        "cells.0.transform_gate.weight",
        "cells.1.linear_all.weight",
    }
    x = torch.randn(4, 2, 3)  # (seq, batch, feature)

    # batch_first module on transposed input must equal the time-first module.
    out_tf, (h_tf, c_tf) = tf(x)
    out_bf, (h_bf, c_bf) = bf(x.transpose(0, 1))
    assert out_bf.shape == (2, 4, 5)
    torch.testing.assert_close(out_bf, out_tf.transpose(0, 1))
    torch.testing.assert_close(h_bf, h_tf)
    torch.testing.assert_close(c_bf, c_tf)

    # Independent unrolled recurrence over both layers.
    assert h_tf.shape == (2, 2, 5)
    assert c_tf.shape == (2, 2, 5)
    torch.testing.assert_close(out_tf[-1], h_tf[-1])
    hs = [torch.zeros(2, 5) for _ in range(2)]
    cs = [torch.zeros(2, 5) for _ in range(2)]
    outs = []
    for t in range(4):
        x_t = x[t]
        for layer, cell in enumerate(tf.cells):
            hs[layer], cs[layer] = cell(x_t, hs[layer], cs[layer])
            x_t = hs[layer]
        outs.append(hs[-1])
    torch.testing.assert_close(out_tf, torch.stack(outs))
    torch.testing.assert_close(h_tf, torch.stack(hs))
    torch.testing.assert_close(c_tf, torch.stack(cs))


def test_state_dtype_and_gradients():
    torch.manual_seed(2)
    model = LSTM_SRU(3, 4, num_layers=2).double()
    x = torch.randn(2, 2, 3, dtype=torch.float64)

    # Explicit state differs from default zero state.
    h0 = torch.randn(2, 2, 4, dtype=torch.float64)
    c0 = torch.randn(2, 2, 4, dtype=torch.float64)
    out_explicit, _ = model(x, (h0, c0))
    out_default, (h, c) = model(x)
    assert not torch.allclose(out_explicit, out_default)

    # Default zero state and outputs follow the input dtype.
    assert out_default.dtype == torch.float64
    assert h.dtype == torch.float64 and c.dtype == torch.float64

    # Gradients reach the input and every parameter through both outputs.
    xd = x.clone().requires_grad_(True)
    out, (h, c) = model(xd)
    (out.sum() + h.sum() + c.pow(2).sum()).backward()
    assert torch.isfinite(xd.grad).all()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
