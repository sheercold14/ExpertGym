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

    def test_response_logprob_details_align_with_summed_api(self):
        import torch

        from opvec.modeling.logprob import response_logprob_details_from_text, response_logprob_from_text

        class Tok:
            def __call__(self, text, add_special_tokens=False):
                vocab = {"a": 0, "b": 1}

                class Result:
                    input_ids = [vocab[token] for token in text.split() if token in vocab]

                return Result()

        class Model(torch.nn.Module):
            def forward(self, input_ids, use_cache=False):
                del use_cache
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2, device=input_ids.device)
                logits[..., 1] = 2.0

                class Output:
                    pass

                out = Output()
                out.logits = logits
                return out

        model = Model()
        summed = response_logprob_from_text(
            torch,
            model,
            Tok(),
            prompt_text="a",
            response_text="b b",
            device="cpu",
            max_length=8,
        )
        details = response_logprob_details_from_text(
            torch,
            model,
            Tok(),
            prompt_text="a",
            response_text="b b",
            device="cpu",
            max_length=8,
        )

        self.assertIsNotNone(details)
        self.assertEqual(details["response_token_ids"], [1, 1])
        self.assertEqual(details["response_mask"], [1, 1])
        self.assertEqual(len(details["old_logprobs"]), 2)
        self.assertAlmostEqual(details["sum_logprob"], summed, places=6)

    def test_response_logprob_details_from_explicit_token_ids(self):
        import torch

        from opvec.modeling.logprob import response_logprob_tensor_details_from_token_ids

        class Tok:
            def __call__(self, text, add_special_tokens=False):
                vocab = {"a": 0, "b": 1, "c": 2}

                class Result:
                    input_ids = [vocab[token] for token in text.split() if token in vocab]

                return Result()

        class Model(torch.nn.Module):
            def forward(self, input_ids, use_cache=False):
                del use_cache
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, device=input_ids.device)
                logits[..., 2] = 3.0

                class Output:
                    pass

                out = Output()
                out.logits = logits
                return out

        details = response_logprob_tensor_details_from_token_ids(
            torch,
            Model(),
            Tok(),
            prompt_text="a b",
            response_token_ids=[2, 2],
            device="cpu",
            max_length=8,
        )

        self.assertIsNotNone(details)
        self.assertEqual(details["response_token_ids"], [2, 2])
        self.assertEqual(details["response_mask"], [1, 1])
        self.assertEqual(int(details["logprobs"].numel()), 2)


if __name__ == "__main__":
    unittest.main()
