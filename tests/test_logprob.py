import unittest


class LogprobTest(unittest.TestCase):
    def test_render_chat_prompt_falls_back_without_template(self):
        from opvec.modeling.logprob import render_chat_prompt

        class Tok:
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("no template")

        text = render_chat_prompt(Tok(), [{"role": "user", "content": "hello"}], "fallback")
        self.assertIn("hello", text)

    def test_response_logprob_tensor_allows_gradients(self):
        import torch

        from opvec.modeling.logprob import response_logprob_tensor_from_text

        class Tok:
            def __call__(self, text, add_special_tokens=False):
                vocab = {"a": 0, "b": 1}

                class Result:
                    input_ids = [vocab[token] for token in text.split() if token in vocab]

                return Result()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_ids, use_cache=False):
                del use_cache
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2, device=input_ids.device)
                logits[..., 1] = self.scale

                class Output:
                    pass

                out = Output()
                out.logits = logits
                return out

        model = Model()
        logp = response_logprob_tensor_from_text(
            torch,
            model,
            Tok(),
            prompt_text="a",
            response_text="b",
            device="cpu",
            max_length=8,
        )
        self.assertIsNotNone(logp)
        (-logp).backward()
        self.assertIsNotNone(model.scale.grad)


if __name__ == "__main__":
    unittest.main()
