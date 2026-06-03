from transformers import GenerationConfig

if __name__ == '__main__':

    myGenerationConfig = GenerationConfig(
        max_length=256,
        do_sample=True,
        early_stopping=True,
        num_beams=4,
        temperature=0.7,
        length_penalty=1.0,
    )

    myGenerationConfig.save_pretrained('./checkpoint/config/generationConfig')