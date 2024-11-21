def get_null(bel_update, bel_prev, y, x, agent, *args, **kwargs):
    return None


def get_updated_mean(bel_update, bel_prev, y, x, agent, *args, **kwargs):
    return bel_update.mean


def get_updated_bel(bel_update, bel_prev, y, x, agent):
    return bel_update


def get_predicted_bel(bel_update, bel_prev, y, x, agent, *args, **kwargs):
    return bel_prev


def get_predicted_mean(bel_update, bel_prev, y, x, agent, *args, **kwargs):
    return bel_prev.mean


def get_prediction_and_error(bel_update, bel_prev, y, x, agent):
    pred = agent.predict(bel_prev, x)
    
    out = {
        "err": pred - y,
        "pred": pred
    }
    return out