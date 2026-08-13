import timm
import torch

from robonav.aqua.model.aqua_resnet18 import AquaResNet18


def test_aqua_resnet18_matches_timm():
    reference = timm.create_model(
        AquaResNet18.model_name,
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3, 4),
    )
    model = AquaResNet18()
    model.load_state_dict(reference.state_dict(), strict=True)
    model.eval()
    reference.eval()

    x = torch.randn(1, 3, 65, 67)
    with torch.no_grad():
        actual = model(x)
        expected = reference(x)

    assert len(actual) == len(expected) == 4
    for actual_feature, expected_feature in zip(actual, expected):
        torch.testing.assert_close(actual_feature, expected_feature, rtol=0, atol=0)
