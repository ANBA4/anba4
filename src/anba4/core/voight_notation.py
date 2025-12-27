"""Voigt notation utilities."""

from dolfin import as_tensor, as_vector


def stressVectorToStressTensor(sv):
    """Transform a contracted stress vector to stress tensor."""
    return as_tensor(
        [[sv[0], sv[5], sv[4]], [sv[5], sv[1], sv[3]], [sv[4], sv[3], sv[2]]]
    )


def stressTensorToStressVector(st):
    """Transform a stress tensor to stress contracted stress vector."""
    return as_vector([st[0, 0], st[1, 1], st[2, 2], st[1, 2], st[0, 2], st[0, 1]])


def stressTensorToParaviewStressVector(st):
    """Transform a stress tensor to stress contracted stress vector."""
    return as_vector([st[0, 0], st[1, 1], st[2, 2], st[0, 1], st[1, 2], st[0, 2]])


def strainVectorToStrainTensor(ev):
    """Transform a engineering strain to strain vector."""
    return as_tensor(
        [
            [ev[0], 0.5 * ev[5], 0.5 * ev[4]],
            [0.5 * ev[5], ev[1], 0.5 * ev[3]],
            [0.5 * ev[4], 0.5 * ev[3], ev[2]],
        ]
    )


def strainTensorToStrainVector(et):
    """Transform a strain tensor to engineering strain."""
    return as_vector(
        [et[0, 0], et[1, 1], et[2, 2], 2.0 * et[1, 2], 2.0 * et[0, 2], 2.0 * et[0, 1]]
    )


def strainTensorToParaviewStrainVector(et):
    """Transform a strain tensor to engineering strain."""
    return as_vector(
        [et[0, 0], et[1, 1], et[2, 2], 2.0 * et[0, 1], 2.0 * et[1, 2], 2.0 * et[0, 2]]
    )
