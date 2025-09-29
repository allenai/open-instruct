import random
import asyncio
import torch
from open_instruct.search_rewards.utils.run_utils import run_litellm_async


USER_PROMPT = """
<context>
{context}
</context>
Now continue the generation to porduce a final answer and wrap it in <answer></answer> tags.
"""

async def generate_partial_rollouts(model_name, messages=None, context=None, system_prompt=None, num_rollouts=1, rollout_length=8192, temperature=1):
    assert messages is not None or context is not None, "Either messages or context must be provided"
    rollouts = []
    for _ in range(num_rollouts):
        if messages is not None:
            rollout = await run_litellm_async(model_name, messages=messages, max_tokens=rollout_length, temperature=temperature)
        else:         
            rollout = await run_litellm_async(model_name, system_prompt=system_prompt, user_prompt=USER_PROMPT.format(context=context), max_tokens=rollout_length, temperature=temperature)
        rollouts.append(rollout)
    return rollouts


async def maybe_replace_on_policy_rollouts_with_partial_rollouts(
    queries,
    decoded_responses, 
    num_samples_per_prompt_rollout, 
    num_rollouts_to_replace_per_prompt, 
    model_name="gpt-5",
    transform_fn_args=None,
    tokenizer=None,
    masks=None,
    responses=None,
    verbose=False,
):
    """
    Among all on-policy rollouts per prompt, randomly substitue num_rollouts_to_replace_per_prompt rollouts' answers with external versions.
    
    Args:
        queries: List of queries
        decoded_responses: List of decoded response strings
        num_samples_per_prompt_rollout: Number of samples per prompt
        num_rollouts_to_replace_per_prompt: Number of rollouts to replace per prompt
        model_name: Name of the external model to use for partial rollouts
        transform_fn_args: Transform function arguments
        tokenizer: Tokenizer for re-tokenizing responses (optional)
        masks: List of masks corresponding to responses (optional)
        responses: List of tokenized responses to update (optional)
    
    Returns:
        If tokenizer, masks, and responses are provided: (modified_responses, updated_masks, updated_responses)
        If tokenizer and masks are provided: (modified_responses, updated_masks)
        Otherwise: modified_responses
    """
    num_prompts = len(decoded_responses) // num_samples_per_prompt_rollout
    
    # Get the system prompt and additional question instructions
    system_prompt = transform_fn_args[0]["system_prompt_text"]
    additional_question_instructions = transform_fn_args[0]["additional_question_instructions"]
    
    # Create a copy of the responses to modify
    modified_responses = decoded_responses.copy()
    new_responses = []
    
    # Process each prompt group
    for i in range(num_prompts):
        start_idx = i * num_samples_per_prompt_rollout
        end_idx = start_idx + num_samples_per_prompt_rollout
        
        # Get the current group of responses for this prompt
        current_group = decoded_responses[start_idx:end_idx]
        current_group_queries = queries[start_idx:end_idx]
        
        # Randomly select indices to replace
        indices_to_replace = random.sample(range(num_samples_per_prompt_rollout), 
                                         min(num_rollouts_to_replace_per_prompt, num_samples_per_prompt_rollout))
        
        # For each selected index, generate a partial rollout replacement
        for local_idx in indices_to_replace:
            global_idx = start_idx + local_idx
            original_response = current_group[local_idx]
            original_query = current_group_queries[local_idx]
            
            # Extract the context before the answer
            context = original_response.split("<answer>")[0]
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": original_query + "\n\n"  #+ additional_question_instructions["long_form"]
                },
                {
                    "role": "assistant",
                    "content": context
                },
                {
                    "role": "user",
                    "content": "Now continue the generation to produce a final answer and wrap it in <answer></answer> tags."
                }
            ]
            
            # Generate partial rollout using the original response as context
            partial_rollouts = await generate_partial_rollouts(
                model_name=model_name,
                messages=messages,
                num_rollouts=1,
                rollout_length=8192,
                temperature=1
            )
            # print("🚼 [Debug] Original query:")
            # print(original_query)
            # print(original_response)
            # print("🚼 [Debug] Partial rollout:")
            # print(partial_rollouts[0])
            
            # Replace the original response with the concatenation of context and partial rollout
            modified_responses[global_idx] = context + partial_rollouts[0]
            new_responses.append(context + partial_rollouts[0])

    # Update masks and responses if tokenizer is provided
    updated_masks = None
    updated_responses = None
    
    if tokenizer is not None and masks is not None:
        updated_masks = masks.copy()
        original_decoded_responses = decoded_responses.copy()
        
        # Also update tokenized responses if provided
        if responses is not None:
            updated_responses = responses.copy()
        
        for i in range(len(modified_responses)):
            if modified_responses[i] != original_decoded_responses[i]:
                # This response was replaced, re-tokenize it
                new_tokens = tokenizer.encode(modified_responses[i], return_tensors="pt").squeeze(0)
                
                # Update the tokenized responses if provided
                if updated_responses is not None:
                    updated_responses[i] = new_tokens
                
                # Update the mask to handle context + new answer concatenation
                # The new response contains: original_context + new_generated_answer
                # The context may contain tool calls (mask=0), so we need to preserve those mask values
                
                # Extract the context (everything before <answer>) from the original response
                original_context = original_decoded_responses[i].split("<answer>")[0] if "<answer>" in original_decoded_responses[i] else ""
                
                if original_context:
                    # Tokenize the original context to determine how many tokens it should have
                    context_tokens = tokenizer.encode(original_context, return_tensors="pt").squeeze(0)
                    context_token_count = len(context_tokens)
                    
                    # Get the original mask and preserve it for the context portion
                    original_mask = updated_masks[i]
                    
                    # The context portion should use the original mask (up to context length)
                    # The new answer portion should be all 1s (since it's generated by external model)
                    if context_token_count <= len(original_mask):
                        # Use original mask for context, pad with 1s for new answer
                        context_mask = original_mask[:context_token_count]
                        new_answer_mask = [1] * (len(new_tokens) - context_token_count)
                        updated_masks[i] = context_mask + new_answer_mask
                    else:
                        # Context is longer than original mask (tokenization difference)
                        # Use original mask + extend with 1s for remaining context + new answer
                        updated_masks[i] = original_mask + [1] * (len(new_tokens) - len(original_mask))
                else:
                    # No context found, treat all tokens as new (shouldn't happen with partial rollouts)
                    updated_masks[i] = [1] * len(new_tokens)
                
                if verbose:
                    # Debug: Check if masked content is preserved and new answers have correct masks
                    print(f"\n🔍 [Mask Validation] Response {i}:")
                    
                    # Check if context was found and analyze mask preservation
                    if original_context:
                        context_tokens = tokenizer.encode(original_context, return_tensors="pt").squeeze(0)
                        context_token_count = len(context_tokens)
                        
                        # Get the original and new context masks
                        original_context_mask = masks[i][:context_token_count] if context_token_count <= len(masks[i]) else masks[i]
                        new_context_mask = updated_masks[i][:context_token_count]
                        
                        # Check if masked content (tool outputs) is preserved
                        masked_positions_original = [j for j, val in enumerate(original_context_mask) if val == 0]
                        masked_positions_new = [j for j, val in enumerate(new_context_mask) if val == 0]
                        
                        masks_preserved = masked_positions_original == masked_positions_new
                        print(f"🛡️  Tool output masks preserved: {'✅ YES' if masks_preserved else '❌ NO'}")
                        
                        if not masks_preserved:
                            print(f"   Original masked positions: {masked_positions_original}")
                            print(f"   New masked positions: {masked_positions_new}")
                        
                        # Check new answer mask (should be all 1s)
                        new_answer_mask = updated_masks[i][context_token_count:]
                        all_ones = all(val == 1 for val in new_answer_mask)
                        print(f"🎯 New answer mask all 1s: {'✅ YES' if all_ones else '❌ NO'}")
                        
                        if not all_ones:
                            non_one_positions = [j for j, val in enumerate(new_answer_mask) if val != 1]
                            print(f"   Non-1 positions in new answer: {non_one_positions[:10]}")
                        
                        # Show summary
                        total_zeros = sum(1 for val in updated_masks[i] if val == 0)
                        total_ones = sum(1 for val in updated_masks[i] if val == 1)
                        print(f"📊 Final mask: {total_zeros} zeros (tool outputs), {total_ones} ones (trainable)")
                        
                    else:
                        # No context found - all should be 1s
                        all_ones = all(val == 1 for val in updated_masks[i])
                        print(f"🎯 All mask values are 1s: {'✅ YES' if all_ones else '❌ NO'} (no context found)")
                    
                    print("─" * 50)
                
    
    # Return appropriate combination based on what was provided
    if updated_responses is not None:
        return modified_responses, updated_masks, updated_responses
    elif updated_masks is not None:
        return modified_responses, updated_masks
    else:
        return modified_responses



if __name__ == "__main__":
    system_prompt = """You are a research assistant who answers questions through iterative reasoning and research.

Process
Use tags to show your reasoning at any point.
Use <call_tool name="...">query</call_tool> when you need information (see tools below).
You can alternate between thinking and searching multiple times.
Only provide <answer></answer> tags when you have enough information for a complete response. If the problem asks for a specific, short-form answer, you can also put the answer string in the \boxed{} format.
Support every non-trivial claim with retrieved evidence. Wrap the exact claim span in <cite id="ID1,ID2">...</cite>, where id are snippet IDs from searched results (comma-separated if multiple). Use only returned snippets; never invent IDs. Avoid citing filler text - cite just the factual claim.
Calling Tools (<call_tool name="...">query</call_tool>)
You can use the following tools:
google_search
Purpose: general web search.
Input via: <call_tool name="google_search">your query</call_tool>
Output: web search snippets (see SEARCH RESULTS).
Optional parameters
gl: geolocation
hl: host language
browse_webpage
Purpose: open a specific URL (typically one returned by google_search) and extract readable page text as snippets.
Input via: <call_tool name="browse_webpage">https://example.com/article</call_tool>
Output: webpage (see SEARCH RESULTS).
snippet_search
Purpose: focused snippet retrieval from scientific papers
Input via: <call_tool name="snippet_search">your query</call_tool>
Output: snippets from existing papers (see SEARCH RESULTS).
Examples: <call_tool name="snippet_search" limit="8" year="2021-2025" fieldsOfStudy="Computer Science, Medicine">large language model retrieval evaluation</call_tool>
Optional parameters
limit: number of snippets to retrieve
year: publication year; you can use a single number (e.g., 2024) or a range (e.g., 2022-2025)
fieldsOfStudy: One or a comma-separated list from: Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics.
Tool Output
After you issue a tool call, we will execute it and return results wrapped in <tool_output> tags.
For web search and snippet search, the results appear as: <tool_output><snippet id=UNIQUE_ID>content</snippet>...</tool_output>
For web browsing, the searched results are represented as <tool_output><webpage id=UNIQUE_ID>content</webpage></tool_output>
Answer and Citation Format
Once you collect all of the necessary information, generate the final answer, and mark your answer with answer tags: <answer></answer>.
If your answer is short (e.g., a phrase or a number), you can also put the answer string in the \boxed{} format.
In your answer, wrap the supported text in <cite id="SNIPPET_ID"> ... </cite>. You have to use the exact ID from a returned <snippet id=...>...</snippet>.
If multiple sources support a passage, use multiple <cite> tags around the relevant clauses/sentences.
Examples
<cite id="S17">LLMs often hallucinate on long-tail facts.</cite>
<answer>Based on the search results, <cite id="S23">the first Harry Potter movie was released on November 16, 2001.</cite>Therefore, the final answer is \boxed{November 16, 2001}.</answer>
WORKFLOW EXAMPLE
Below is a simple example that demonstrates the process and the correct use of tools and tags. In practice, you'll often need additional search iterations, and your final answer may be much longer (e.g., a multi-paragraph report).

Question: Give a concise update on 2024 renewable energy market trends and current commercial solar efficiency benchmarks.

<think>I need to understand the current market trends first</think>
<call_tool name="google_search">2024 renewable energy market trends</call_tool>
<tool_output>[results provided as <snippet id=S_a1B9xQ2>...</snippet>, <snippet id=S_p0Zr41Q>...</snippet>]</tool_output>

<think>Now I need specific data on solar panel efficiency</think>
<call_tool name="snippet_search" limit="5" year="2023-2025" fieldsOfStudy="Engineering, Materials Science">latest solar panel efficiency 2024</call_tool>
<tool_output>[results provided as <snippet id=S_x4xU7dU>...</snippet>, <snippet id=S_GxA2ZLh>...</snippet>]</tool_output>

<think>I have enough to answer succinctly</think>
<answer>
Global renewables expanded rapidly in 2024, <cite ids="S_p0Zr41Q,S_GxA2ZLh">driven primarily by the growth of solar and wind energy.</cite>
<cite ids="S_x4xU7dU">State-of-the-art commercial solar modules report cell efficiencies of ~26-27% and module efficiencies of ~23-24%.</cite>
\boxed{Solar leads 2024 renewables; top commercial module efficiency ~ 23-24%}
</answer>

REQUIREMENTS
Think and search iteratively until you have sufficient information
Only provide the final answer when ready
Cite all claims from search results using exact snippet IDs"""

    question = """
    Has microwave-assisted fermentation of feather hydrolysate with brewer's spent grain been explored for co-production of peptides and bioenergy? Find prior work and reserach gap

For the given question, please write a comprehensive, evidence-backed answers to scientific questions. You should ground every nontrivial claim in retrieved snippets. Cite using <cite id="...">...</cite> drawn only from returned snippets. Please prefer authoritative sources (peer-reviewed papers, reputable benchmarks/docs) and prioritize recent work for fast-moving areas. You should acknowledge uncertainty and conflicts; if evidence is thin or sources disagree, state it and explain what additional evidence would resolve it. It's important to structure with clear markdown headers and a coherent flow. In each section, write 2-5 sentence paragraphs with clear topic sentences and transitions; use lists sparingly only when they improve clarity. Ideally, you should synthesize rather than enumerate content: it's helpful to group findings across papers, explain relationships, and build a coherent narrative that answers the question, supported by citations. Most importantly, DO NOT invent snippets or citations and never fabricate content.
    """
    search_context = """<call_tool name="snippet_search" limit="10" year="2018-2025" fieldsOfStudy="Environmental Science,Engineering,Biology,Chemistry">
feather hydrolysate fermentation peptides keratinolytic Bacillus hydrolysate peptides
</call_tool>
<tool_output>
<snippet id="S_HSe1bSr">
Title: Keratinases as Versatile Enzymatic Tools for Sustainable Development
Text: Three Bacillus strains (CL18, CL33A, CL14) were evaluated for their keratinolytic activity in converting chicken feathers into protein hydrolysates through submerged fermentation in feather broth medium. The fermentation process occurred at 30°C and pH 7 for varying durations, with Bacillus sp. CL18 demonstrating the highest efficiency, degrading 98.9% of feathers in 7 days at a concentration of 10% feathers. Bacillus sp. CL33A and CL14 showed lower degradation rates of 56.2% and 71.4%, respectively. The resulting hydrolysates exhibited significant bioactive properties, including antioxidant activities and inhibition of dipeptidyl peptidase IV (DPP IV) and angiotensin I-converting enzyme (ACE), with Bacillus sp. CL18 achieving 95.7% DPP IV and 89.7% ACE inhibition after 5 days in a higher feather concentration. These findings suggest that the protein hydrolysates from Bacillus sp. CL18 fermentation could serve as a valuable source of bioactive peptides for nutritional and pharmaceutical applications. Additionally, keratinase from Bacillus sp. CL18 was partially purified and used to convert various protein substrates into bioactive hydrolysates under optimal conditions.</snippet>
<snippet id="S_LjIK9EX">
Title: Keratinases as Versatile Enzymatic Tools for Sustainable Development
Text: Technologies for bioactive peptides production include (1) natural source extraction, (2) genetic engineering, (3) cell-free expression systems, (4) transgenic animals and plants and (5) chemical and enzymatic synthesis [137,138,[140][141][142][143][144]. Event though, most of the bioactive peptides were identified in food and food byproducts [136,139], waste biomass can also become a source of molecules with bioactive properties [34,37,133,142,145]. Extraction from natural sources is a commonly used strategy for releasing and obtaining peptides from food proteins, based on (a) in vitro enzymatic hydrolysis, (b) cultivation of protein-degrading microorganisms (fermentation) and (c) animal digestive enzymes hydrolysis [136,138]. For in vitro food-protein hydrolysis, proteases with broad substrate specificity, from plant (ficin, bromelain, papain), animal (pepsin, chymotrypsin, trypsin) and microbial (proteinase K, collagenase, subtilisin A) sources, are needed [136]. As the representatives of proteolytic enzymes, keratinases can be potentially used for bioactive peptides production via enzymatic hydrolysis of keratins and other proteins, or by full cell fermentation of microbial media supplemented with poultry feather and non-keratin plant proteins [68,142,145].

Several studies, exploring the possibility of obtaining bioactive hydrolysates and peptides from keratin waste have been conducted. Three previously evaluated Bacillus strains of keratinolytic activities, CL18, CL33A and CL14, were used for full-cell bioconversion of chicken feathers into protein hydrolysates [133] via submerged cultivation in feather broth (FB) medium (0.5 g/L NaCl, 0.3 g/L K 2 HPO 4 , and 0.4 g/L KH 2 PO 4 ) with whole chicken feathers (10, 20, 30, 40 and 50 g/L).</snippet>
<snippet id="S_yT1w4fZ">
Title: Keratinases as Versatile Enzymatic Tools for Sustainable Development
Text: Pseudomonas sp. P5 has been studied for its ability to valorize poultry feathers into protein hydrolysates, which can be used as microbial media. The bacterium was cultured on mineral medium with varying concentrations of raw wet feathers (30 to 90 g/L). After five days, feather hydrolysis rates ranged from 70% to 93%, with the highest rates observed at lower feather concentrations. At 70 g/L, the keratinolytic activity reached a maximum of 36.2 ± 8.4 U/mL. The study compared the peptide and amino acid content of hydrolysates from full-cell fermentation with those from enzymatic hydrolysis (using partially purified keratinase) and weak alkaline hydrolysis. The results showed that the full-cell fermentation produced lower amino acid concentrations (301.2 mg/L at 90 g/L and 274.8 mg/L at 70 g/L) compared to enzymatic hydrolysis (1047.6 mg/L at 90 g/L and 1090.7 mg/L at 70 g/L). However, the peptide concentrations were higher in the full-cell fermentation hydrolysates (6.2 g/L and 4.6 g/L) compared to the enzymatic hydrolysates (3.2 g/L and 3.3 g/L).</snippet>
<snippet id="S_CHwWwyV">
Title: The keratinolytic bacteria Bacillus cytotoxicus as a source of novel proteases and feather protein hydrolysates with antioxidant activities
Text: The keratinolytic bacteria Bacillus cytotoxicus as a source of novel proteases and feather protein hydrolysates with antioxidant activities</snippet>
<snippet id="S_OF9abyn">
Title: Insights into the keratin efficient degradation mechanism mediated by Bacillus sp. CN2 based on integrating functional degradomics
Text: The study analyzed feather hydrolysates from Bacillus sp. CN2 using HPLC, revealing that soluble peptides began to appear in the fermentation broth after 6 hours of proteolysis, with significant increases in both content and types of peptides as feather degradation progressed. By 24 hours, 86.70% of the feathers were degraded, and 10 notable peptide peaks were identified after 72 hours of fermentation. The molecular weight of the soluble peptides ranged from 166 to 1043 Da, indicating effective keratin degradation by the proteases secreted by strain CN2. The research also examined free amino acids produced during fermentation, noting that rapid feather degradation within the first 24 hours led to low concentrations of free amino acids, primarily due to the action of S8B serine endopeptidases and T3 γ-glutamyltransferase. After 36 hours, M4 metalloendopeptidase further hydrolyzed keratin, yielding hydrophobic amino acids such as methionine, phenylalanine, leucine, isoleucine, and tyrosine, which align with the hydrophobic amino acid composition of feathers.</snippet>
<snippet id="S_SIBhe9x">
Title: Insights into the keratin efficient degradation mechanism mediated by Bacillus sp. CN2 based on integrating functional degradomics
Text: Bacillus sp. CN2 effectively degrades chicken feather keratin, a challenging substrate due to its recalcitrance. The strain demonstrated significant keratin degradation, achieving an 86.70% weight loss within 24 hours and producing 195.05 U/mL of keratinases at 48 hours, along with 0.40 mg/mL of soluble proteins at 60 hours. The extracellular protease consortium exhibited broad substrate specificity and biodegradability towards both soluble and insoluble proteins. The degradation mechanism involves T3 γ-glutamyltransferase, which reduces the disulfide bonds in keratin, followed by S8B serine endopeptidases that hydrolyze keratin to reveal more cleavage sites. Subsequently, keratin is broken down into small peptides through the action of additional proteases, including M4, S8C, and S8A. High-performance liquid chromatography (HPLC) and amino acid analysis confirmed the presence of numerous soluble peptides and essential amino acids in the hydrolysate. The findings highlight the critical roles of specific proteases in the keratin degradation process by Bacillus sp. CN2.</snippet>
<snippet id="S_VQBW9Zz">
Title: Biotransformation of keratin waste to amino acids and active peptides based on cell-free catalysis
Text: Therefore, continued optimization of fermentation parameters and feeding strategies will have the opportunity to significantly reduce cell division time, thereby reducing the entire time required for a fermentation cycle.

Although this study achieved rapid degradation of feathers via keratinase, the catalytic mechanism employed by keratinase has not yet been characterized, accounting for one of the primary issues faced by keratinase researchers [44]. Similarly, an abundance of polypeptides is present in the hydrolysate; however, the specific functions of these peptides are unknown. However, the custom production of functional peptides cannot be achieved by keratinase hydrolysis of feathers. Therefore, research on the substrate specificity and catalytic mechanism of keratinase will be of top priority for future studies.</snippet>
<snippet id="S_Pz6uXpN">
Title: Chicken Feather Waste Valorization Into Nutritive Protein Hydrolysate: Role of Novel Thermostable Keratinase From Bacillus pacificus RSA27
Text: Microbial keratinases exhibit a momentous role in converting keratin biowastes into exceedingly valuable protein supplements. This study reports a novel, highly stable keratinase from Bacillus pacificus RSA27 for the production of pure peptides rich in essential amino acids from chicken feathers. Purified keratinase showed a specific activity of 38.73 U/mg, 2.58-fold purification, and molecular weight of 36 kDa. Kinetic studies using a chicken feather as substrate report Km and Vmax values of 5.69 mg/ml and 142.40 μg/ml/min, respectively, suggesting significant enzyme-substrate affinity/biocatalysis. Identification and in silico structural-functional analysis of keratinase discovered the presence of distinct amino acid residues and their positions. Besides, keratinase possesses a high-affinity calcium-binding site (Asp128, Leu162, Asn164, Ile166, and Val168) and a catalytic triad of Asp119, His151, and Ser308, known attributes of serine protease (subtilisin family). Furthermore, a scale-up to 5 L fermenter revealed complete feather hydrolysis (94.5%) within 24 h with high activity (789 U/ml) and total amino acid of 153.97 μmol/ml. Finally, cytotoxicity evaluation of protein hydrolysate resulted in negligible cytotoxic effects (1.02%) on the mammalian hepatoblastoma cell line, signifying its potential biotechnological applications.</snippet>
<snippet id="S_MYu66Rc">
Title: Biotransformation of keratin waste to amino acids and active peptides based on cell-free catalysis
Text: This study efficiently expressed KerZ1 in B. subtilis WB600 by applying rational design at the transcriptional, translational, and fermentation levels allowing for the enzymatic activity of KerZ1 to reach 426.60 KU/mL. We also propose a method for degrading feathers (100 g/L) to produce amino acids based on the synergistic effect of high-activity KerZ1 and sodium sulfite, with the conversion rate reaching 56.6%. In addition, this method is easy to use, quick, and does not generate toxic byproducts; it is also conducive to industrial scale-up production. In future studies, we plan to focus on the active peptides in feather hydrolysate, which will serve as a new, inexpensive source of functional peptides.</snippet>
<snippet id="S_ArND2M1">
Title: Novel keratinolytic enzymes, discovered from a talented and efficient bacterial keratin degrader
Text: Huge quantities of keratinaceous waste are a substantial and almost totally unexploited protein resource which could be upgraded for use as high value-added products by efficient keratinolytic enzymes. In this study, we found that Bacillus sp. 8A6 can efficiently degrade chicken feather after 24 h growth. According to phylogenetic analysis, the strain (formerly identified as Bacillus pumilus 8A6) belongs to the B. pumilus species clade but it is more closely related to B. safensis. Hotpep predicted 233 putative proteases from Bacillus sp. 8A6 genome. Proteomic analysis of culture broths from Bacillus sp. 8A6 cultured on chicken feathers or on a mixture of bristles and hooves showed high abundance of proteins with functions related to peptidase activity. Five proteases (one from family M12, one from family S01A, two from family S08A and one from family T3) and four oligopeptide and dipeptide binding proteins were highly expressed when Bacillus sp. 8A6 was grown in keratin media compared to LB medium. This study is the first to report that bacterial proteases in families M12, S01A and T3 are involved in keratin degradation together with proteases from family S08.</snippet>
</tool_output>
<think>
Recap and plan: The goal is to determine whether "microwave-assisted fermentation of feather hydrolysate with brewer's spent grain (BSG) for co-production of peptides and bioenergy" has prior art, and identify the research gap. I began by gathering evidence on feather hydrolysate/keratinolytic fermentation producing peptides. I now need to (1) establish BSG's use in bioenergy fermentation/anaerobic digestion, (2) assess microwave-assisted steps: microwave hydrolysis/pretreatment of feathers, microwave pretreatment of BSG, or microwave-assisted fermentation/dark fermentation/AD, and (3) find any studies combining feather hydrolysate and BSG, with or without microwave assistance, especially co-production of peptides and energy. If no direct match exists, I'll triangulate the closest adjacent works to outline the gap.

Summary of first query: The initial snippets confirmed that microbial keratinolytic fermentation of feathers produces peptide-rich hydrolysates with bioactivities; several Bacillus strains achieve high feather degradation and produce peptides in hydrolysates, sometimes with antioxidant or ACE/DPP-IV inhibitory activities. They provide quantitative evidence (e.g., near-complete degradation within 24–48 h with certain strains; peptide and amino acid yields; molecular weight ranges). However, there is no mention yet of BSG, bioenergy, or microwave assistance.

Gap → Next query: I need evidence on BSG in bioenergy contexts and any microwave pretreatment. Exact next query: "brewer's spent grain anaerobic digestion biohydrogen microwave pretreatment" (2018–2025; Environmental Science/Engineering). This will surface whether BSG is commonly used for biogas/hydrogen and whether microwave steps are studied, helping assess feasibility and prior art intersections.
</think>
<call_tool name="snippet_search" limit="10" year="2018-2025" fieldsOfStudy="Environmental Science,Engineering,Biology,Chemistry">
brewer's spent grain anaerobic digestion biohydrogen microwave pretreatment
</call_tool>
<tool_output>
<snippet id="S_xxbLbou">
Title: Brewer's Spent Grains: Possibilities of Valorization, a Review
Text: Brewer's Spent Grains: Possibilities of Valorization, a Review</snippet>
<snippet id="S_k9D0nNF">
Title: Brewer's Spent Grains—Valuable Beer Industry By-Product
Text: The brewing sector is a significant part of the global food industry. Breweries produce large quantities of wastes, including wastewater and brewer's spent grains. Currently, upcycling of food industry by-products is one of the principles of the circular economy. The aim of this review is to present possible ways to utilize common solid by-product from the brewing sector. Brewer's spent grains (BSG) is a good material for sorption and processing into activated carbon. Another way to utilize spent grains is to use them as a fuel in raw form, after hydrothermal carbonization or as a feedstock for anaerobic digestion. The mentioned by-products may also be utilized in animal and human nutrition. Moreover, BSG is a waste rich in various substances that may be extracted for further utilization. It is likely that, in upcoming years, brewer's spent grains will not be considered as a by-product, but as a desirable raw material for various branches of industry.</snippet>
<snippet id="S_AOjJyNz">
Title: Brewer's Spent Grains: Possibilities of Valorization, a Review
Text: This review was based on updated research on how to use brewer's spent grains (BSG). The use of BSG was considered both in food, as an ingredient or using value-added components derived from brewer's spent grain, or in non-food products such as pharmaceuticals, cosmetics, construction, or food packaging. BSG is a valuable source of individual components due to its high nutritional value and low cost that is worth exploiting more to reduce food waste but also to improve human health and the environment. From the bioeconomy point of view, biological resources are transformed into bioenergetically viable and economically valuable products. The pretreatment stage of BSG biomass plays an important role in the efficiency of the extraction process and the yield obtained. The pretreatments presented in this review are both conventional and modern extraction methods, such as solvent extractions or microwave-assisted extractions, ultrasonic-assisted extractions, etc.</snippet>
<snippet id="S_o7TNNqb">
Title: Brewer's spent grain as a no-cost substrate for polyhydroxyalkanoates production: assessment of pretreatment strategies and different bacterial strains.
Text: Brewer's spent grain as a no-cost substrate for polyhydroxyalkanoates production: assessment of pretreatment strategies and different bacterial strains.</snippet>
<snippet id="S_Gz6iUKO">
Title: Suitability of Composting Process for the Disposal and Valorization of Brewer's Spent Grain
Text: Suitability of Composting Process for the Disposal and Valorization of Brewer's Spent Grain</snippet>
<snippet id="S_dydlWgX">
Title: Recycling Potential of Brewer's Spent Grains for Circular Biorefineries
Text: Recycling Potential of Brewer's Spent Grains for Circular Biorefineries</snippet>
<snippet id="S_QSBQ4F4">
Title: Hydrothermal Carbonization of Brewery's Spent Grains for the Production of Solid Biofuels
Text: To make a beer there are four essential ingredients needed: water, malt, hops, and yeast. After brewing process, the main wastes are spent grains. These are often used as additions to fodders in animal husbandry. This study presents preliminary results of an investigation aiming to determine the feasibility of an alternative use of spent grains as a potential source of solid fuel. This source of energy could make breweries partly sustainable in terms of their energy supply. Such an approach may be feasible especially in large scale industrial breweries. This preliminary study presents encouraging results, showing improvements in terms of the fuel properties of the spent grain after its valorization through hydrothermal carbonization. Moreover, qualitative GC-MS analysis also indicates potential feasibility of the liquid byproduct of the hydrothermal carbonization of spent grain for biogas production. Results of proximate, ultimate, and DTG analyses show that hydrothermal carbonization of spent grain could improve its fuel properties and make it an especially suitable feedstock for fast pyrolysis and gasification. Improvement of HHV is also an improvement in terms of combustion.</snippet>
<snippet id="S_2P80Pt8">
Title: Brewer's spent grain as a no-cost substrate for polyhydroxyalkanoates production: assessment of pretreatment strategies and different bacterial strains.
Text: Physicochemical characterization of the brewer's spent grain used.</snippet>
<snippet id="S_9regZTh">
Title: By-Products in the Malting and Brewing Industries—Re-Usage Possibilities
Text: Brewer's spent grain, a by-product of the malting and brewing industries, can be repurposed to produce biofuels, including biohydrogen, instead of being discarded as waste. Various microorganisms have been utilized for bioethanol production through fermentation. While the direct production of biogas from spent grain was historically challenging due to the complexity of its lignocellulosic structure, recent advancements in thermal-chemical and microwave pretreatment methods have enhanced its digestibility and biogas yield. Research indicates that anaerobic digestion of spent grain, particularly in a bioreactor setting, shows significant potential for biogas production. Additionally, spent grain is a valuable source of phenolic compounds, which possess antioxidant properties, and can be extracted using modern techniques such as ultrasound and microwave processing. Protein extraction from spent grain has also been successful, particularly through hydrothermal pretreatment combined with enzymatic methods. Overall, these findings highlight the potential for spent grain to contribute to sustainable biofuel production and the extraction of valuable compounds.</snippet>
<snippet id="S_ODYrjBn">
Title: Evaluating the potential applications of brewers' spent grain in biogas generation, food and biotechnology industry: A review
Text: Evaluating the potential applications of brewers' spent grain in biogas generation, food and biotechnology industry: A review</snippet>
</tool_output>
<think>
Recap of plan: The question asks whether "microwave-assisted fermentation of feather hydrolysate with brewer's spent grain (BSG) for co-production of peptides and bioenergy" has been previously explored, and to identify the research gap. I will triangulate four evidence pillars: (1) feather hydrolysis/fermentation to peptide-rich hydrolysates, (2) BSG valorization for bioenergy (biogas/biohydrogen) and as fermentation co-substrate, (3) microwave-assisted steps applied to feathers and to BSG (pretreatment, hydrolysis, or fermentation/digestion), and (4) any studies that combine feather-derived hydrolysate with BSG, particularly with microwave assistance, and report co-production of peptides and energy.

Stage so far: From the first query, I have solid evidence that keratinolytic fermentation of feathers produces peptide-rich hydrolysates with bioactivities, but nothing yet on BSG or microwaves. The second query shows BSG is widely valorized for biogas and other products, and that microwave pretreatment has been explored to enhance digestibility and biogas yields.

What's missing: I need explicit evidence on microwave-assisted processing for feathers (hydrolysis/pretreatment and any "microwave-assisted fermentation"), and whether feather wastes have been co-digested or co-fermented with BSG (with or without microwaves). If no direct match, I will articulate the closest reported cascades to define the gap.

Next query: Search for microwave-assisted hydrolysis/pretreatment of feathers/keratin to produce hydrolysates (2015–2025; Environmental Science/Chemistry/Engineering). This helps establish whether microwave is established for feather hydrolysate production.
</think>
<call_tool name="snippet_search" limit="10" year="2015-2025" fieldsOfStudy="Environmental Science,Engineering,Chemistry,Biology">
microwave-assisted hydrolysis feathers keratin hydrolysate microwave pretreatment feathers peptides
</call_tool>
<tool_output>
<snippet id="S_M8FZyQh">
Title: Closing the Loop with Keratin-Rich Fibrous Materials
Text: Microwave technology has been developed in the last years as an alternative heating method. The advantage of microwave is a reduced reacting time and an important energetic economy since heating is faster due to the internal heat generation and homogeneous rise of temperature for all the polar molecules present in the reactor [36]. Microwave-assisted heating has been applied by several authors for keratin extraction from different biomasses. Zoccola et al. [64] used microwave treatment to solubilize keratin from wool. In their work microwave irradiation with power ranging between 150-570 W was applied for 60 min to wool samples, at different temperatures (150; 170; 180 • C), and a maximum extraction yield of 62% was obtained at 180 • C. Extracted protein fraction was characterized by a very low molecular mass, among 3-8 kDa but the major drawback associated with this technique was that higher temperature-induced significant cystine loss which increased up to 99% at 180 • C. Lee et al. [127] combined microwave treatment and alkali hydrolysis for the extraction of keratin from feathers and considered it more efficient compared to the conventional heating-alkali method. It was reported that the maximum extraction yield of around 26 mg/mL protein was obtained by using a power of 800 W for 10 min, and 0.5 M sodium hydroxide by a liquor ratio of 1:50 [127].</snippet>
<snippet id="S_cV01MCl">
Title: Valorization of Livestock Keratin Waste: Application in Agricultural Fields
Text: Microwave-assisted hydrolysis of livestock keratin waste, particularly feathers, involves using strong acid or alkali solutions for treatment, followed by hydrolysis, neutralization, and drying. Common agents include HCl, H2SO4, Ca(OH)2, KOH, and NaOH. The process breaks peptide and disulfide bonds, affecting the yield and molecular weight of keratin. Longer hydrolysis times yield more soluble keratin but result in lower molecular weights, while shorter times increase molecular weight but decrease yield. Research indicates that microwave-alkali treatment produces a higher concentration of amino acids (69.4 mg/g of feathers) compared to autoclave-alkali (19.0 mg/g) and conventional heating-alkali (27.8 mg/g) methods. However, acid-alkali treatment generates significant alkaline wastewater and waste acid vapor, posing environmental risks. To enhance efficiency and reduce energy consumption, alternative methods like acoustic cavitation-assisted alkaline hydrolysis have been explored, as they can operate at room temperature and low pressure. The choice of treatment method and hydrolysis duration should align with the intended application of the keratin products.</snippet>
<snippet id="S_Wy8fl99">
Title: Sustainable Management of Keratin Waste Biomass: Applications and Future Perspectives
Text: Feather hydrolysate obtained by chemicalthermal hydrolysis of chicken feather is rich in amino acids and polypeptides (Coward-Kelly et al. 2006). Due to similar composition with soybean protein and cotton seed protein the hydrolysate used as diet supplement for feeding ruminants.

Enzymatic-alkaline hydrolysis of feather keratin for feed was done by (Dalev 1994;Dalev et al. 1996). Enzymatic modification through enrichment with lysine leads to increase in the feed nutritive value (Dalev et al. 1997) Raw horns and hoofs are put on high pressure in a rendering plant to make horn meal. Keratin is a useful protein which is helpful to prepare the animal feed (Brandelli et al. 2015).</snippet>
<snippet id="S_zdxmfny">
Title: Valorization of Livestock Keratin Waste: Application in Agricultural Fields
Text: The most significant disadvantage of acid-alkali treatment is the large amount of alkaline wastewater and waste acid vapor produced during the process, which endangers both equipment and the environment. To save energy and improve the extraction rate, this method has been combined with other methods. The acoustic cavitation-assisted alkaline hydrolysis of wool is an energy-saving process, compared to steam-assisted alkaline hydrolysis, because it can be performed at room temperature and at low pressure [22]. Protein hydrolysate from microwave-alkali treatment contains a significantly higher concentration of amino acids (69.4 mg•g −1 of feathers) than the protein hydrolysate of the autoclave-alkali (19.0 mg•g −1 of feathers) and conventional heating-alkali (27.8 mg•g −1 of feathers) treatments [23].</snippet>
<snippet id="S_lq9JX1s">
Title: Materials diversity of hydrogel: Synthesis, polymerization process and soil conditioning properties in agricultural field
Text: The chicken feather is abundant by-products of Poultry farms and these feathers are chemically protein in nature that consists of keratin protein majorly. The keratin is contained a higher content of amino acid cysteine (7%) than present in canola (0.63%) and soy (1.26%). The researchers have focused to utilize these waste products of poultry farms in fruitful form for agricultural benefits. The feather keratin hydrogel is based on a protein solution of hydrolyzed feather keratin (hydrolyzed by 1 M hydrochloric acid) with subsequent addition of two initiators (potassium persulfate (KPS) and sodium bisulfate;SBS), N, N methylene bisacrylamide and neutralized acrylic acid. The stirring continues till the end of the polymerization process then hydrogel washed by ethanol (two times). The maximum water absorbency (501 g/g) in water has achieved at the optimum composition. The utilization of HFK in hydrogel makes it biodegradable and eco-friendly with significant properties required for a soil conditioner. It will be more eco-friendly with the environment if keratin hydrolysis takes place under microwave radiations or superheated steam. This hydrogel is very sensitive to a minute change in pH and saline concentration [117].</snippet>
<snippet id="S_MhEHM3w">
Title: Recent progress in the conversion of biomass wastes into functional materials for value-added applications
Text: Keratin, a key component in feathers, hair, and other animal by-products, is produced in large quantities globally, particularly in poultry and textile industries. It exists as α-keratin in hair and β-keratin in feathers, contributing to the mechanical durability of materials. Hydrolysis methods, including alkaline, acid, and enzyme treatments, are commonly used to extract soluble keratin, with yields influenced by factors such as temperature, pH, and treatment duration. For instance, a study reported a 94% yield of keratin from chicken feathers using 2.5% NaOH pretreatment combined with thermo-chemical treatments. Other methods, such as using sodium metabisulfite, achieved yields up to 96%, but faced challenges like high operational costs and environmental concerns. An alternative two-step thermal hydrolysis process developed by Tasaki yielded approximately 70% keratin from hog hair without chemical use, highlighting a potential for more sustainable extraction methods.</snippet>
<snippet id="S_QxZ4ADE">
Title: Bioconversion of Keratin Wastes Using Keratinolytic Microorganisms to Generate Value-Added Products
Text: Keratin hydrolysates from feather waste, which is composed of 90% keratin, are increasingly recognized as a valuable protein source for animal feed and dietary supplements. The recycling of keratinous waste is essential for improving its digestibility and nutritional value. To achieve this, keratin materials must undergo hydrolysis, often facilitated by keratinolytic microorganisms or their enzymes, which convert keratin into smaller peptides and amino acids. This biological treatment significantly enhances digestibility, with treated keratin materials achieving up to 98% digestibility compared to only 2% for untreated ones. Studies indicate that incorporating keratin hydrolysate supplements into animal feed can improve its nutritional profile. For example, the strain Brevibacillus sp. AS-S10-II produces an alkaline keratinase that effectively converts feather keratin into essential amino acids, making feather hydrolysates a promising candidate for protein supplementation in animal feed formulations.</snippet>
<snippet id="S_aO2B9bv">
Title: Synthesis and Characterization of Methyl Cellulose/Keratin Hydrolysate Composite Membranes
Text: Commercially available cleaned and degreased mixtures of feathers and down in a mixing ratio of 70/30 of the domestic goose (Anser anser domesticus)-as used for pillow fillings-were employed as starting material. For preparation of the keratin hydrolysate solutions, feathers were solubilized in deionized water as described below. Methyl cellulose (viscosity 400 cP, 2% in H 2 O, 27.5%-31.5% methoxyl) and glycerol (purity 99%) both from Sigma-Aldrich (Schnelldorf, Germany) were used as received without further purification.</snippet>
<snippet id="S_iudPTfA">
Title: Biotransformation of keratin waste to amino acids and active peptides based on cell-free catalysis
Text: Analysis of the types of amino acids as well as the content of the feather hydrolysate (0.12 g) generated via synergism of KerZ1 and 1% sodium sulfite is shown in Table 2. Complete hydrolysis of the feathers resulted in amino acid production with a conversion rate of 49.3%. Among them, the contents of glutamic acid, alanine, tyrosine, phenylalanine, leucine, and lysine were relatively high.

To maximize the conversion rate of the product, we optimized the amount and shape (feather or feather meal) of feathers, and the results show that when the amount of feathers was 100 g/L, the conversion rate of amino acids reached a maximum of 56.6% (Fig. 5a, b). Moreover, the wide variety of amino acids demonstrates the enormous potential of hydrolysates as feed protein additives (Fig. 5c). The hydrolysate also contains a mixed short peptide with a molecular weight of approximately 1.3 kDa (Fig. 5e), and mass spectrometry results revealed up to 12 active peptides that exhibited catalytic activity and antioxidant activity (Fig. 5d, f ).</snippet>
<snippet id="S_gbjyDMy">
Title: Effect of ultrasound on keratin valorization from chicken feather waste: Process optimization and keratin characterization
Text: The effect of different ultrasonic time and power on feather hydrolysis was evaluated by assaying soluble peptides (Fig. 3A and B). The result showed that ultrasonic time remarkably improved peptides content growing from about 29 mg⋅mL − 1 at 2 h to 389 mg⋅mL − 1 at10 h, while ultrasonic power made less effect with peptides content growing from about 207 mg⋅mL − 1 at 100 W to 283 mg⋅mL − 1 at 600 W for 4 h. Fig. 3A indicated that there was about 47 % of them with Mw less than 3 kDa at 2 h while the peptides < 0.5 kDa was about 8.7 %. As processing time was prolonged to 10 h, around 65 % of peptides < 3 kDa were produced, among which about 20 % were smaller than 0.5 kDa. Similarly, power rising also resulted in the proportion increasement of smaller peptides, which contributed to feather dissolution (Fig. 3B). The result illustrated that this process effectively facilitated the breakage of feather keratin. Despite the beneficial effect of ultrasound on feather degradation, the quality of regenerated keratin should also be concerned. The influence of various ultrasonic power and time on keratin properties solubility and amino acids composition will be further discussed.</snippet>
</tool_output>
<think>
Recap: The question asks whether "microwave-assisted fermentation of feather hydrolysate with brewer's spent grain (BSG) for co-production of peptides and bioenergy" has prior art, and what the research gap is. I'm assembling four evidence pillars: (1) fermentation/hydrolysis of feathers to peptide-rich hydrolysates; (2) BSG use in bioenergy and as a fermentation substrate; (3) microwave-assisted processing applied to feathers/BSG and to fermentation/digestion; and (4) any studies that combine feather hydrolysate with BSG, particularly with microwaves, and report co-production of peptides and energy.

So far:

Pillar (1) is well supported: multiple keratinolytic strains convert feathers to peptide-rich hydrolysates with bioactivities and quantifiable peptide profiles/yields, in submerged fermentation; peptide MW ranges and activity (ACE/DPP-IV inhibition) are reported. However, no mention of BSG, energy, or microwaves in those snippets.
Pillar (2) shows BSG is a common bioenergy feedstock for anaerobic digestion/biohydrogen and that pretreatments including microwave can enhance digestibility/biogas, per reviews.
Pillar (3) is supported for feathers: microwave-assisted alkali hydrolysis of feathers/wool to keratin/hydrolysate is reported, with quantitative yield improvements vs conventional methods; but not yet for "microwave-assisted fermentation" per se.
Missing: Evidence that (a) feather hydrolysate is used for bioenergy (biogas/hydrogen) via fermentation/dark fermentation/AD, (b) co-digestion/fermentation of feather waste or feather hydrolysate with BSG, and (c) the specific combination "microwave-assisted fermentation of feather hydrolysate with BSG" for "co-production of peptides and bioenergy." I will now search for anaerobic digestion/dark fermentation using feather waste/hydrolysate and for co-digestion with BSG. Precise next query: "feather hydrolysate anaerobic digestion biohydrogen dark fermentation feathers biogas" (2015–2025; Environmental Science/Engineering/Biology).
</think>
<call_tool name="snippet_search" limit="10" year="2015-2025" fieldsOfStudy="Environmental Science,Engineering,Biology">
feather hydrolysate anaerobic digestion biohydrogen dark fermentation feathers biogas
</call_tool>
<tool_output>
<snippet id="S_e6Bixbs">
Title: Microbial Keratinases: Enzymes with Promising Biotechnological Applications
Text: Anaerobic digestion is an effective method for producing renewable energy from waste substrates, including feathers. During this process, feathers are hydrolyzed into amino acids, which are then converted into organic acids, ammonia, carbon dioxide, hydrogen, and minor sulfur compounds, serving as substrates for methane production. The theoretical methane yield from proteins is approximately 0.496 m³/kg of volatile solids, but actual yields from waste feathers are around 0.21 m³/kg due to the compact keratin structure that limits nutrient accessibility. Pretreatment methods, including chemical, physical, and enzymatic approaches, are being explored to enhance this yield. Notably, a recombinant strain of Bacillus megaterium, which carries the keratinase gene from Bacillus licheniformis, has been developed to improve feather degradation. This strain has shown the ability to effectively hydrolyze feathers, leading to methane production of up to 80% of the theoretical yield from protein digestion, attributed to its efficient enzyme production.</snippet>
<snippet id="S_LAEjdVf">
Title: Valorization of Livestock Keratin Waste: Application in Agricultural Fields
Text: Research indicates that livestock keratin waste, particularly from feathers, can be effectively converted into biogas through anaerobic digestion. Dried feathers exhibit a methane potential of 0.05 m³/kg wet weight, which can be enhanced through various pretreatment methods. For instance, pretreatment with recombinant Bacillus megaterium has achieved methane production of 0.35 Nm³/kg, reaching 80% of the theoretical protein value. Other pretreatments, including those using Bacillus sp. C4, Ca(OH)₂, thermal, and enzymatic methods, have also significantly increased biogas yields from feathers. Additionally, feather hydrolysate, derived from Streptomyces sampsonii GS1322, is rich in amino acids and proteins, and has been shown to improve soil quality. Its application on barren agricultural land resulted in increased wheat germination rates and plant height, as well as a notable rise in beneficial soil microorganisms while reducing pathogenic fungi. Thus, feather hydrolysate serves as a low-cost soil amendment, enhancing agricultural productivity.</snippet>
<snippet id="S_P3FZA20">
Title: The keratinolytic bacteria Bacillus cytotoxicus as a source of novel proteases and feather protein hydrolysates with antioxidant activities
Text: The poultry industry generates significant amounts of feather waste, primarily composed of keratin, which poses disposal challenges due to its resistance to degradation. Traditional disposal methods, such as burning, are environmentally harmful. An alternative approach involves microbial degradation, which utilizes keratinolytic bacteria to hydrolyze feathers, producing enzymes like keratinases and proteases, as well as protein hydrolysates rich in bioactive peptides. This eco-friendly method aligns with green practices and enhances the digestibility of feather protein, making it suitable for various industrial applications, including detergents, leather, textiles, and medical uses. The increasing interest in converting feather waste into valuable bioproducts highlights the need for further research into enzyme systems that can effectively transform organic waste into bioactive peptides. While several proteases have been identified from thermophilic bacteria, few are known for their keratinolytic activity.</snippet>
<snippet id="S_w84tcTa">
Title: Identification of two new keratinolytic proteases from a Bacillus pumilus strain using protein analysis and gene sequencing
Text: The global poultry processing industry generates approximately 8.5 million tons of feather waste annually, which is often disposed of through dumping, incineration, or landfilling, leading to environmental issues. While feathers can be converted into feather meal for animal feed, regulatory constraints and high treatment costs limit this use. An alternative method for managing feather waste is anaerobic digestion, which can produce renewable energy in the form of methane and yield digested residues that can be safely used as fertilizer due to the eradication of pathogens during the process. Keratin, the primary component of feathers, is a challenging material to degrade, with feathers primarily composed of β-keratin. Various microorganisms, particularly those from the Bacillus genus, have been identified as capable of producing keratinases that can facilitate the breakdown of keratinous materials.</snippet>
<snippet id="S_rY94Qgj">
Title: Keratinases as Versatile Enzymatic Tools for Sustainable Development
Text: The bacterium Pseudomonas sp. P5 has been studied for its ability to convert poultry feathers into protein hydrolysates, which can be used as microbial media. The strain demonstrated significant proteolytic and keratinolytic activity when cultured on mineral medium with varying concentrations of raw wet feathers (30 g/L to 90 g/L). After five days, hydrolysis rates ranged from 70% to 93%, with the highest rates observed at lower feather concentrations. Increased substrate levels (70 and 90 g/L) led to greater bacterial biomass and enzyme activity, achieving a maximum keratinolytic activity of 36.2 ± 8.4 U/mL at 70 g/L. Comparisons of the final peptide and amino acid content from full-cell fermentation versus enzymatic hydrolysis showed that the fermentation resulted in lower amino acid concentrations (301.2 mg/L for 90 g/L and 274.8 mg/L for 70 g/L) compared to enzymatic hydrolysis (1047.6 mg/L for 90 g/L and 1090.7 mg/L for 70 g/L). However, the fermentation produced slightly higher peptide concentrations (6.2 g/L and 4.6 g/L) compared to enzymatic hydrolysis (3.2 g/L and 3.3 g/L).</snippet>
<snippet id="S_AIfj17k">
Title: Sustainable Management of Keratin Waste Biomass: Applications and Future Perspectives
Text: Feather hydrolysate obtained by chemicalthermal hydrolysis of chicken feather is rich in amino acids and polypeptides (Coward-Kelly et al. 2006). Due to similar composition with soybean protein and cotton seed protein the hydrolysate used as diet supplement for feeding ruminants.

Enzymatic-alkaline hydrolysis of feather keratin for feed was done by (Dalev 1994;Dalev et al. 1996). Enzymatic modification through enrichment with lysine leads to increase in the feed nutritive value (Dalev et al. 1997) Raw horns and hoofs are put on high pressure in a rendering plant to make horn meal. Keratin is a useful protein which is helpful to prepare the animal feed (Brandelli et al. 2015).</snippet>
<snippet id="S_NOUWClZ">
Title: Feather hydrolysate as a promising nitrogen-rich fertilizer for greenhouse lettuce cultivation
Text: Feathers bioconversion was performed through submerged cultivations using the keratinolytic bacterium Bacillus sp. CL18. Bacterial suspensions prepared in sterile saline were employed (1 mL) to inoculate Erlenmeyer flasks (250 mL) containing 50 mL of feather broth, composed of mineral medium (MM; 0.5 g/L NaCl, 0.3 g/L K 2 HPO 4 , 0.4 g/L KH 2 PO 4 ) and chicken feathers (30.0 g/L). Initial pH was adjusted to 7.5 before autoclaving (Sobucki et al. 2017).

Incubations were performed at 30 °C, 125 rpm, for 7 days. After this period the media were filtered, the filtrates were autoclaved (121 °C, 105 kPa, 15 min) and centrifuged (9500 × g for 10 min). Collected supernatants were employed as feather hydrolysate (FH). Partial chemical characterization of FH, performed as described above, is presented in Table 1.</snippet>
<snippet id="S_D477kbu">
Title: Biotransformation of keratin waste to amino acids and active peptides based on cell-free catalysis
Text: Analysis of the types of amino acids as well as the content of the feather hydrolysate (0.12 g) generated via synergism of KerZ1 and 1% sodium sulfite is shown in Table 2. Complete hydrolysis of the feathers resulted in amino acid production with a conversion rate of 49.3%. Among them, the contents of glutamic acid, alanine, tyrosine, phenylalanine, leucine, and lysine were relatively high.

To maximize the conversion rate of the product, we optimized the amount and shape (feather or feather meal) of feathers, and the results show that when the amount of feathers was 100 g/L, the conversion rate of amino acids reached a maximum of 56.6% (Fig. 5a, b). Moreover, the wide variety of amino acids demonstrates the enormous potential of hydrolysates as feed protein additives (Fig. 5c). The hydrolysate also contains a mixed short peptide with a molecular weight of approximately 1.3 kDa (Fig. 5e), and mass spectrometry results revealed up to 12 active peptides that exhibited catalytic activity and antioxidant activity (Fig. 5d, f ).</snippet>
<snippet id="S_n7XiEne">
Title: Progress in Microbial Degradation of Feather Waste
Text: Feathers are a major by-product of the poultry industry. They are mainly composed of keratins which have wide applications in different fields. Due to the increasing production of feathers from poultry industries, the untreated feathers could become pollutants because of their resistance to protease degradation. Feathers are rich in amino acids, which makes them a valuable source for fertilizer and animal feeds. Numerous bacteria and fungi exhibited capabilities to degrade chicken feathers by secreting enzymes such as keratinases, and accumulated evidence shows that feather-containing wastes can be converted into value-added products. This review summarizes recent progress in microbial degradation of feathers, structures of keratinases, feather application, and microorganisms that are able to secrete keratinase. In addition, the enzymes critical for keratin degradation and their mechanism of action are discussed. We also proposed the strategy that can be utilized for feather degradation. Based on the accumulated studies, microbial degradation of feathers has great potential to convert them into various products such as biofertilizer and animal feeds.</snippet>
<snippet id="S_4q4op0w">
Title: Effective biodegradation of chicken feather waste by co-cultivation of keratinase producing strains
Text: BackgroundChicken feather, a byproduct of poultry-processing industries, are considered a potential high-quality protein supplement owing to their crude protein content of more than 85%. Nonetheless, chicken feathers have been classified as waste because of the lack of effective recycling methods. In our previous studies, Bacillus licheniformis BBE11-1 and Stenotrophomonas maltophilia BBE11-1 have been shown to have feather-degrading capabilities in the qualitative phase. To efficiently recycle chicken feather waste, in this study, we investigated the characteristics of feather degradation by B. licheniformis BBE11-1 and S. maltophilia BBE11-1. In addition, in an analysis of the respective advantages of the two degradation systems, cocultivation was found to improve the efficiency of chicken feather waste degradation.ResultsB. licheniformis BBE11-1 and S. maltophilia BBE11-1 were used to degrade 50 g/L chicken feather waste in batches, and the degradation rates were 35.4% and 22.8% in 96 h, respectively. The degradation rate of the coculture system reached 55.2% because of higher keratinase and protease activities. Furthermore, cocultivation was conducted in a 3 L fermenter by integrating dissolved oxygen control and a two-stage temperature control strategy. Thus, the degradation rate was greatly increased to 81.8%, and the conversion rate was 70.0% in 48 h. The hydrolysates exhibited antioxidant activity and contained large quantities of amino acids (895.89 mg/L) and soluble peptides.ConclusionsCocultivation of B. licheniformis BBE11-1 and S. maltophilia BBE11-1 can efficiently degrade 50 g/L chicken feather waste and produce large amounts of amino acids and antioxidant substances at a conversion rate of 70.0%.</snippet>
</tool_output>
<think>
I need to determine whether "microwave-assisted fermentation of feather hydrolysate with brewer's spent grain (BSG) for co-production of peptides and bioenergy" has prior art and to articulate the research gap. I will build evidence across four pillars: (1) feather keratin hydrolysis/fermentation yielding peptide-rich hydrolysates; (2) BSG use in fermentation/anaerobic digestion for bioenergy and its pretreatments; (3) microwave-assisted processing applied to feathers and/or BSG (particularly pretreatments that enhance hydrolysis or digestion); and (4) any direct combinations of feather hydrolysate with BSG, especially with microwave assistance, and any cascades that co-produce peptides and energy. I have already gathered evidence for (1) keratinolytic fermentations yielding bioactive peptides; (2) BSG valorization for bioenergy and that microwave pretreatment is used; and (3) microwave-assisted hydrolysis of keratin/feathers. What remains is to see if anyone has co-digested/fermented feather waste or feather hydrolysate with BSG or similar lignocellulosic residues, and whether any study explicitly frames a cascade for co-production of peptides and bioenergy using microwave assistance. I will synthesize the findings and identify the specific gap if no direct precedent exists. For the answer, I will structure it into: background on feather-to-peptides, BSG-to-bioenergy and microwave pretreatments, microwave-assisted keratin hydrolysis, evidence on feathers in AD/bioenergy, and an integration/gap section that assesses whether the precise combination has been reported and what remains to be done.
</think>"""
    sample_context =  question + "\n\n" + search_context
    rollouts = asyncio.run(generate_partial_rollouts(model_name="gpt-5", context=sample_context, system_prompt=system_prompt, num_rollouts=1, rollout_length=8192, temperature=1))
    print(rollouts[0])
