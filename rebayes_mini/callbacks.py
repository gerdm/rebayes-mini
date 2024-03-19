def get_null(bel_update, bel_prev, y, x, *args, **kwargs):
    return None


def get_updated_mean(bel_update, bel_prev, y, x):
    return bel_update.mean


def get_updated_bel(bel_update, bel_prev, y, x):
    return bel_update


def get_predicted_mean(bel_update, bel_prev, y, x):
    return bel_prev.mean
