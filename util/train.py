from .all import *


def train_step(x, y, model, optimizer, loss_fn, train_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_metric.update_states(y, logits)

    return loss_value


def test_step(x, y, model, loss_fn, val_metric):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_metric.update_states(y, loss_value)

    return loss_value


def train(
    train_set, val_set, model, loss_fn, optimizer, train_metric, val_metric, args
):
    print("Training Start")
    train_loss = []
    val_loss = []
    for epoch in range(args.epochs):
        for step, (x, y) in enumerate(train_set):
            loss_value = train_step(x, y, model, optimizer, loss_fn, train_metric)
            train_loss.append(loss_value)

        for step, (x, y) in enumerate(val_set):
            loss_value = test_step(x, y, model, loss_fn, val_metric)
            val_loss.append(loss_value)

        train_metric_result = train_metric.result()
        val_metric_result = val_metric.result()

        print(
            f"[{epoch + 1} / {args.epochs}], loss:{train_loss[epoch]:.4f}, mse: {float(train_metric_result):.4f}, "
            f"val_loss: {val_loss[epoch]:.4f}, val_mse:{float(val_metric_result):.4f}"
        )

        train_metric.reset_states()
        val_metric.reset_states()

    train_plot(train_loss, val_loss)


def forecasting(x, model, scale=None, inverse=False):
    true = []
    predicted = []
    for n, (X, y) in enumerate(x):
        true.append(y.numpy())
        temp = model.predict(X.numpy())
        predicted.append(temp)

    true = np.array(true)
    predicted = np.array(predicted)

    if inverse:
        if scale is None or not scale:
            raise "scale is not define"
        true = scale.inverse_transform(true.reshape(-1, 1))
        predicted = scale.inverse_transform(predicted.reshape(-1, 1))

    return true, predicted
