import logging
import torch
import torch.nn.functional as F
import os


def train(model, train_reader, dev_reader, model_dir, model_name):
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    epoch = 10
    step = 0
    best_step = 0
    log_every_k = 10

    early_patience = 10

    best_res = 0

    for epoch in range(epoch):
        input, labels = train_reader.read_batch()
        optimizer.zero_grad()

        logits = model(input)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        optimizer.step()
        step += 1

        # Eval on dev.
        if not step % log_every_k:
            dev_res = eval(dev_reader)

            if dev_res > best_res:
                best_res = dev_res
                best_step = step
                torch.save(model, os.path.join(model_dir, model_name))
            else:
                if step - best_step > early_patience:
                    logging.info(
                        "Early stop with patience %d." % early_patience
                    )

    train_reader.tag_vocab.fix()
