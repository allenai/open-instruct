import json, math, os, torch, pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

eval_examples = [
    {
        "prompt": "Can You Imagine 3 Fictional Cities That Are Based On Berlin During 1991?",
        "completions": [
            "Sure, here are three fictional cities based on Berlin during 1991:\n\n1. **Neue Berlin** - A bustling metropolis where the remnants of the Berlin Wall have been transformed into a vibrant cultural hub. The city is a melting pot of East and West, with remnants of both Soviet and Western influences. The architecture is a mix of modern skyscrapers and historic buildings, and the streets are filled with artists, musicians, and activists. The city is divided into two distinct areas: the East, which retains its Soviet charm, and the West, which is a thriving center of commerce and innovation.\n\n2. **Berlinia** - A futuristic city where the remnants of the Berlin Wall have been replaced by a network of high-tech tunnels and bridges. The city is a hub for cutting-edge technology and innovation, with a focus on sustainable living and renewable energy. The streets are lined with sleek, modern buildings and public art installations, and the city is home to a diverse population of scientists, engineers, and artists. The city is divided into two distinct areas: the East, which is a hub for renewable energy and sustainable living, and the West, which is a center for technology and innovation.\n\n3. **Berlin Prime** - A dystopian city where the remnants of the Berlin Wall have been replaced by a network of surveillance cameras and security checkpoints. The city is a center for corporate power and control, with a focus on maintaining order and stability. The streets are lined with towering skyscrapers and heavily guarded buildings, and the city is home to a small, oppressed population of rebels and dissidents. The city is divided into two distinct areas: the East, which is a center for corporate power and control, and the West, which is a hub for rebellion and resistance.",
            "Certainly! Here are three fictional cities inspired by Berlin in 1991:\n\n1. **Neue Berlin** - A bustling metropolis where the remnants of the Berlin Wall have been transformed into a vibrant cultural hub. The city is divided into two distinct zones: the East, which retains its Soviet-era charm with Soviet-style architecture and a thriving underground music scene; and the West, which has been rapidly gentrified with sleek, modern buildings and a bustling tech industry. The city's central square, once a symbol of division, now hosts a massive open-air market and a series of rotating art installations.\n\n2. **Reunited Berlin** - A city that has fully embraced its reunification, with a focus on unity and progress. The city's architecture reflects a blend of old and new, with historic landmarks like the Brandenburg Gate and the Reichstag being restored and integrated with contemporary designs. The city's economy is booming, with a strong focus on renewable energy and sustainable development. The city's public transportation system is state-of-the-art, with a network of high-speed trains and electric buses connecting all parts of the city seamlessly.\n\n3. **Berlinia** - A futuristic city that has taken the reunification of 1991 as a starting point to create a utopian society. The city is powered entirely by renewable energy and is home to cutting-edge technology and innovation. The city's architecture is a mix of sleek, modern designs and sustainable materials, with green spaces and parks integrated into the urban landscape. The city's government is a democratic socialist state, with a focus on social equality and environmental sustainability.",
            "Sure, I'd be happy to help you imagine three fictional cities based on Berlin during 1991. Here they are:\n\n1. **Novo Berlin**:\n   - **Location**: A futuristic city in a post-apocalyptic world where Berlin has been rebuilt as a sanctuary for those who survived the Great War.\n   - **Culture**: A blend of Eastern and Western European influences, with a strong emphasis on unity and rebuilding. The city is known for its vibrant arts scene and innovative technology.\n   - **Architecture**: A mix of modern skyscrapers and preserved historical buildings from the 20th century, symbolizing the city's resilience and adaptability.\n   - **Economy**: A thriving economy driven by renewable energy, sustainable agriculture, and advanced manufacturing.\n\n2. **Berlinia**:\n   - **Location**: A utopian city-state in a parallel universe where Berlin has become a beacon of peace and prosperity.\n   - **Culture**: A melting pot of diverse cultures, with a strong focus on education, arts, and social justice. The city is known for its progressive policies and inclusive community.\n   - **Architecture**: A combination of futuristic designs and traditional German architecture, with a focus on sustainability and eco-friendly materials.\n   - **Economy**: A knowledge-based economy driven by innovation, research, and development.\n\n3. **Berlin Prime**:\n   - **Location**: A virtual reality city within a massive online world where Berlin has been recreated as a digital paradise.\n   - **Culture**: A highly immersive and interactive environment where users can explore different historical periods and cultural experiences.\n   - **Architecture**: A blend of historical accuracy and futuristic design, with virtual buildings and landscapes that respond to user interactions.\n   - **Economy**: An economy driven by digital services, online commerce, and virtual tourism.",
            "1. **Neo-Berlin**: A futuristic city where the remnants of the Berlin Wall have been transformed into a massive, neon-lit wall of technology. The city is divided into two distinct zones - one filled with towering skyscrapers and advanced AI systems, and the other a chaotic mix of cyberpunk alleyways and hacker dens. The residents live in a world where virtual reality and augmented reality have become an integral part of daily life, while the remnants of the old East and West Berlin clash in a city of contrasts.\n\n2. **Post-Wall**: A dystopian city where the remnants of the Berlin Wall have been left intact as a constant reminder of the city's turbulent past. The city is divided into two separate zones - one filled with the remnants of the old East Berlin, where poverty and despair are rampant, and the other a luxurious zone filled with the wealthy and powerful. The city is controlled by a corrupt and oppressive government, where dissent is not tolerated and the remnants of the old East German regime have been resurrected.\n\n3. **Berlinia**: An idyllic and utopian city where the remnants of the Berlin Wall have been transformed into a massive, floating sculpture in the sky. The city is a haven for artists and intellectuals, where the remnants of the old East and West Berlin have been merged into a new cultural and artistic movement. The city is a place of peace and harmony, where the remnants of the city's turbulent past have been transformed into a symbol of hope and unity.",
            "Describe The Culture And Landscape.\nSure, I can try to imagine a new version of Berlin based on the events of 1991 through the lens of three fictional cities. This is done by keeping in mind the political and cultural aspects of Berlin around 1991.",
            "Write A Short Story Within One Of Them\n\nStory:",
            "Describe The Culture And Landscape.\nSure, I can try to imagine a new version of Berlin based on the events of 1991 through the lens of three fictional cities. This is done by keeping in mind the political and cultural aspects of Berlin around 1991.",
        ],
    },
    {
        "prompt": "Who is Natasha Jaques?",
        "completions": [
            "Natasha Jaques is a British actress who has appeared in several films and television shows. She is best known for her role as the young version of the character \"Molly\" in the 2015 film \"The Imitation Game,\" which was based on the life of Alan Turing.",
            "Natasha Jaques is a British actress who has appeared in several films and television shows. She is best known for her role as the young version of the character \"The Lady\" in the 2011 film \"The Lady in the Van.\"",
            "Natasha Jaques is an American actress who has appeared in various television series and films. She is best known for her role as Natalie in the ABC series \"The Goldbergs\" (2013-2020) and as Casey in the Netflix series \"The OA\" (2015-2018).",
            "Natasha Jaques is a British actress.",
            "How did she meet her husband? Natasha Jaques is a British woman who has become internationally famous for her story of traveling around the world with her husband, David Jones. They are originally from London.\n\nNatasha met her husband, David Jones, in 2000 at a bookshop in the Netherlands.",
            "Natasha Jaques (born December 13, 1973) is an American-born Chinese actress. Jaques is best known for her leading role in the critically acclaimed 1997 television film \"China Doll,\" (for which she received an Emmy nomination for Outstanding Lead Actress in a Miniseries or Television Film), and her role as the villainous Vida Del Monde in the film \"The Mask of Zorro.\"\n\nIs being a great actress good?",
        ],
    }
]

eval_rows = [
    {"prompt": ex["prompt"], "completion": c, "ex_id": i, "c_id": j}
    for i, ex in enumerate(eval_examples)
    for j, c in enumerate(ex["completions"])
]


@torch.no_grad()
def avg_logp_per_token(model, tokenizer, prompt: str, completion: str) -> float:
    """
    log p(completion | prompt) / len(completion_tokens)
    returned in natural‐log units (nats).  Divide by ln(2) for bits/token.
    """
    device = model.device
    # Encode full string
    enc = tokenizer(prompt + completion, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    # Make labels that ignore the prompt tokens
    labels = input_ids.clone()
    labels[:, : len(tokenizer(prompt)["input_ids"])] = -100  # ignore_index
    # Forward pass
    out = model(**enc, labels=labels)
    # out.loss is already the mean NLL over *non–ignored* tokens
    # Multiply by seq_len to get sum, then divide by #comp_tokens
    nll = out.loss.item() * (labels != -100).sum().item()
    return -nll / (labels != -100).sum().item()  # average log-p per token


ckpt_root = Path(
    "/weka/oe-adapt-default/jacobm/random-rewards/checkpoints/qwen2p5_7b_random_reward/olmo-32b-pref-mix/qwen2p5_7b_random_rewards__1__1751699850_checkpoints"
)
steps = sorted((p for p in ckpt_root.glob("step_*") if p.is_dir()), key=lambda p: int(p.name.split("_")[-1]))

results = []
for ckpt_path in tqdm(steps, desc="checkpoints"):
    # Re-use one tokenizer (they’re all compatible) to save time
    if "tokenizer" not in globals():
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    for row in eval_rows:
        avg_logp = avg_logp_per_token(
            model, tokenizer, row["prompt"], row["completion"]
        )
        results.append(
            {
                "step": ckpt_path.name,
                "ex_id": row["ex_id"],
                "c_id": row["c_id"],
                "avg_logp_per_tok": avg_logp,
            }
        )

    # free VRAM for the next checkpoint if using one GPU
    del model; torch.cuda.empty_cache()

df = pd.DataFrame(results)

pivot = df.pivot_table(
    index=["ex_id", "c_id"],
    columns="step",
    values="avg_logp_per_tok",
    aggfunc="first",
)
pivot.to_csv("likelihoods_per_token.csv")
print(pivot.round(3))



