def qp_qs_to_qkappa_qmu(vp, vs, Qp, Qs):
    """
    Перевод Qp, Qs -> Qkappa, Qmu
    для SPECFEM2D, plane strain.

    Parameters
    ----------
    vp : float
        Скорость P-волны.
    vs : float
        Скорость S-волны.
    Qp : float
        Добротность P-волны.
    Qs : float
        Добротность S-волны.

    Returns
    -------
    Qkappa : float
        Добротность объемной части.
    Qmu : float
        Добротность сдвиговой части.
    """

    Qmu = Qs

    r = (vs / vp) ** 2

    denominator = 1.0 / Qp - r / Qmu

    if denominator <= 0:
        raise ValueError(
            "Невозможно получить положительное Qkappa: "
            f"denominator = {denominator}"
        )

    Qkappa = (1.0 - r) / denominator

    return Qkappa, Qmu


if __name__ == "__main__":
    vp = 3297.849
    vs = 2222.536

    Qp = 21.19305151645515
    Qs = 20.0

    Qkappa, Qmu = qp_qs_to_qkappa_qmu(
        vp=vp,
        vs=vs,
        Qp=Qp,
        Qs=Qs,
    )

    print(f"QKappa = {Qkappa:.15g}")
    print(f"Qmu = {Qmu:.15g}")
