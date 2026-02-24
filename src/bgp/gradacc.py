from typing import Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.cuda.amp.grad_scaler import GradScaler

class GradientAccumulation:
    def __init__(
        self,
        actual_batch_size: int,     # micro-batch size (e.g., 3)
        expect_batch_size: int,     # effective batch size (e.g., 27)
        loader_len: int,
        optimizer: Optimizer,
        grad_scaler: Optional[GradScaler] = None,
        max_norm: Optional[float] = None,
    ) -> None:
        assert expect_batch_size % actual_batch_size == 0, \
            "expect_batch_size must be divisible by actual_batch_size"

        self.actual_batch_size = actual_batch_size
        self.expect_batch_size = expect_batch_size
        self.loader_len = loader_len
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.max_norm = max_norm

        self.steps_until_update = expect_batch_size // actual_batch_size  # int

    def step(self, loss: Tensor, step: int) -> None:
        # For mean-style losses: average gradients across accumulation steps
        loss = loss / self.steps_until_update

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % self.steps_until_update == 0 or (step + 1) == self.loader_len:
            if self.grad_scaler is not None:
                if self.max_norm is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                    self._clip_gradients()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                if self.max_norm is not None:
                    self._clip_gradients()
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

    def _clip_gradients(self) -> None:
        params = [p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params, self.max_norm)
