__all__ = ["train_step", "validataion_step", "train", "predict_step"]

from .all import *


def train_step(x, y, model, optimizer, loss_fn, train_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_metric.update_state(y, logits)

    return loss_value


def validataion_step(x, y, model, loss_fn, val_metric):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_metric.update_state(y, loss_value)

    return loss_value


def predict_step(x, model):
    test_logits = model(x, training=False)
    return test_logits


def train(
    train_set, val_set, model, loss_fn, optimizer, train_metric, val_metric, args
):
    print("Training Start")
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        train_steps = 0
        val_steps = 0
        for step, (x, y) in enumerate(train_set):
            loss_value = train_step(x, y, model, optimizer, loss_fn, train_metric)
            train_loss += loss_value
            train_steps = step + 1

        for step, (x, y) in enumerate(val_set):
            loss_value = validataion_step(x, y, model, loss_fn, val_metric)
            val_loss += loss_value
            val_steps = step + 1

        train_losses.append(train_loss / train_steps)
        val_losses.append(val_loss / val_steps)

        train_metric_result = train_metric.result()
        val_metric_result = val_metric.result()

        end = time.time()
        print(
            f"[{epoch + 1} / {args.epochs}], ",
            f"{end - start:.2f} sec",
            f"loss:{train_loss[epoch]:.4f}, ",
            f"mse: {float(train_metric_result):.4f}, ",
            f"val_loss: {val_loss[epoch]:.4f}, ",
            f"val_mse:{float(val_metric_result):.4f}",
        )

        train_metric.reset_states()
        val_metric.reset_states()

    train_plot(train_losses, val_losses)
