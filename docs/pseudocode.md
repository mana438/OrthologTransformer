# OrthologTransformer: Pseudocode

This document provides a simplified overview of OrthologTransformer with species vector prepended to input sequences.

---

## Training

**Inputs:**
- Tokenized sequence pairs (X, Y)
- Species labels (S_src, S_tgt)
- Model, loss function, optimizer, epochs E

**Pseudocode:**

```
For epoch = 1 to E:
    For each (x, y, s_src, s_tgt) in batches:
        v_src = EmbedSpecies(s_src)
        v_tgt = EmbedSpecies(s_tgt)

        x_input = Concat(v_src, x)  # Prepend species vector to input
        y_input = Concat(v_tgt, y)  # Prepend species vector to target (optional)

        z = Encoder(x_input)
        y_hat = Decoder(z)

        loss = CrossEntropy(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Inference

**Inputs:**
- Tokenized input x
- Source and target species (s_src, s_tgt)
- Trained model, max output length L

**Pseudocode:**

```
v_src = EmbedSpecies(s_src)
v_tgt = EmbedSpecies(s_tgt)

x_input = Concat(v_src, x)

z = Encoder(x_input)

output = [<BOS>]
For t = 1 to L:
    next_token = DecoderStep(z, output, v_tgt)
    Append next_token to output
    If next_token == <EOS>: break

Return output
```

---
