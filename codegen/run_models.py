from codegen.infer import run_model


def run_models():
    models = [
        "NeelNanda/SoLU_1L512W_C4_Code",
        # "NeelNanda/SoLU_2L512W_C4_Code",
        # "NeelNanda/SoLU_3L512W_C4_Code",
        # "NeelNanda/SoLU_4L512W_C4_Code",
        # "NeelNanda/SoLU_6L768W_C4_Code",
        # "NeelNanda/SoLU_8L1024W_C4_Code",
        # "NeelNanda/SoLU_10L1280W_C4_Code",
        # "NeelNanda/SoLU_12L1536W_C4_Code",
        # "NeelNanda/GELU_1L512W_C4_Code",
        # "NeelNanda/GELU_2L512W_C4_Code",
        # "NeelNanda/GELU_3L512W_C4_Code",
        # "NeelNanda/GELU_4L512W_C4_Code",
        "NeelNanda/Attn_Only_1L512W_C4_Code",
        # "NeelNanda/Attn_Only_2L512W_C4_Code",
        # "NeelNanda/Attn_Only_3L512W_C4_Code",
        # "NeelNanda/Attn_Only_4L512W_C4_Code",
        # "gpt2",
        # "gpt2-medium"
    ]

    results = {}
    for model_name in models:
        results["model"] = run_model(model_name)


run_models()
