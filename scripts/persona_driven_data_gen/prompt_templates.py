math_template = """Create a math problem related to the following persona:

{persona}

Note:

1. The math problem should be challenging and involve advanced mathematical skills and knowledge. Only top talents can solve it correctly.
2. You should make full use of the persona description to create the math problem to ensure that the math problem is unique and specific to the persona.
3. Your response should always start with "Math problem:". Your response should not include a solution to the created math problem.
4. Your created math problem should include no more than 2 sub-problems.
"""

math_template_easy = """Create a grade school math word problem related to the following persona:

{persona}

Note:

1. You should make full use of the persona description to create the math problem to ensure that the math problem is unique and specific to the persona.
2. The problem primarily involves performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach a single final answer.
3. Your response should always start with "Math problem:". Your response should not include a solution to the created math problem.
"""


instruction_template = '''Guess a prompt that the following persona may ask you to do:

{persona}

Note:

1. The prompt should be informative and specific.
2. Your output should start with "User prompt:"'''

knowledge_template = '''{persona}

Assume you are the persona described above and you are writing a Quora article using your knowledge, skills, experience, or insights to help others learn and benefit from it.

Note:

1. The article should be specific, informative and knowledge-rich.
2. Your response should start with "Title:"'''

npc_template = """World of Warcraft (WoW) is a massively multiplayer online role-playing game (MMORPG) developed by Blizzard Entertainment. It is set in the high-fantasy world of Azeroth, a land filled with rich lore, diverse races, and epic conflicts. The game has evolved significantly since its release in 2004, with numerous expansions adding new continents, races, classes, and storylines. Below is a detailed overview of the game's worldview, story background, and some key characters and NPCs.

### Worldview and Story Background

**Azeroth** is a world steeped in ancient history, powerful magic, and epic conflicts. The planet is divided into several continents, each with its own unique environments, cultures, and histories. The main continents include:

- Eastern Kingdoms: Home to the human kingdoms, dwarves, gnomes, and the undead Forsaken.
- Kalimdor: Inhabited by orcs, night elves, tauren, trolls, and other races.
- Northrend: A frozen continent, home to the Lich King and the undead Scourge.
- Pandaria: A mystical land shrouded in mists, home to the Pandaren.
- Broken Isles: The remnants of the ancient Night Elf civilization and the site of the Tomb of Sargeras.
- Zandalar and Kul Tiras: Introduced in the Battle for Azeroth expansion, these are the homelands of the Zandalari trolls and the human kingdom of Kul Tiras, respectively.
- Shadowlands: The realm of the afterlife, introduced in the Shadowlands expansion.

The story of Azeroth is vast and complex, spanning millennia and involving numerous races, factions, and cosmic forces. Here are some key aspects of the world's background:

#### **The Titans and the Old Gods**

- **The Titans**: Azeroth was shaped by the Titans, colossal beings who are part of the Pantheon, a group of god-like entities dedicated to bringing order to the universe. The Titans discovered Azeroth and found it infested with chaotic entities known as the Old Gods. To combat this, they created the Titan-forged, including the Keepers, to help shape and protect the world.

- **The Old Gods**: These malevolent, ancient beings sought to corrupt Azeroth. The Titans imprisoned the Old Gods beneath the surface of the world, but their influence persisted, causing chaos and corruption throughout history. Notable Old Gods include C'Thun, Yogg-Saron, N'Zoth, and Y'Shaarj.

#### **The Sundering**

- **The Well of Eternity**: At the center of ancient Kalimdor was the Well of Eternity, a source of immense arcane power. The Highborne, a group of night elves led by Queen Azshara, recklessly tapped into its power, attracting the attention of the Burning Legion, a demonic army led by the dark titan Sargeras.

- **The War of the Ancients**: This conflict saw the night elves, dragons, and other races unite to repel the Burning Legion's invasion. The war culminated in the Sundering, a catastrophic event that shattered the supercontinent of Kalimdor into several smaller continents and created the Maelstrom, a massive, swirling vortex of energy.

#### **The Rise and Fall of Empires**

- **The Troll Empires**: Before the Sundering, the trolls established powerful empires, such as the Gurubashi and Amani. These empires declined over time but left a lasting impact on Azeroth's history.

- **The Night Elf Empire**: After the Sundering, the night elves established a new empire, centered around the World Tree, Nordrassil. They became the guardians of nature and the Emerald Dream, a parallel realm of primal life.

- **The Human Kingdoms**: Humans emerged as a dominant race in the Eastern Kingdoms, founding powerful kingdoms such as Stormwind, Lordaeron, and Dalaran. These kingdoms played crucial roles in the defense of Azeroth against various threats.

#### **The First and Second Wars**

- **The First War**: The orcs, originally from the world of Draenor, were corrupted by the Burning Legion and transported to Azeroth through the Dark Portal. They waged war against the human kingdom of Stormwind, ultimately destroying it.

- **The Second War**: The orcs, now united under the Horde, continued their conquest, clashing with the Alliance of Lordaeron, a coalition of human, dwarf, and high elf forces. The Alliance eventually triumphed, and the orcs were interned in camps.

#### **The Scourge and the Lich King**

- **The Lich King**: Created by the demon lord Kil'jaeden, the Lich King was originally the orc shaman Ner'zhul. He was transformed into a powerful undead entity and imprisoned in the Frozen Throne in Northrend. The Lich King created the Scourge, an army of undead, to pave the way for a new invasion by the Burning Legion.

- **The Third War**: The Scourge ravaged the human kingdoms, leading to the fall of Lordaeron and the rise of the undead Forsaken. The war culminated in the Battle of Mount Hyjal, where the combined forces of the night elves, Horde, and Alliance defeated the Burning Legion.

#### **The Burning Crusade and Beyond**

- **The Burning Crusade**: The first expansion of WoW saw players journey to Outland, the shattered remnants of Draenor, to combat the Burning Legion and its allies.

- **Wrath of the Lich King**: This expansion focused on the conflict with the Lich King in Northrend, culminating in his defeat at Icecrown Citadel.

- **Cataclysm**: The return of the corrupted Dragon Aspect Deathwing caused massive upheaval across Azeroth, reshaping the world and leading to new conflicts.

- **Mists of Pandaria**: This expansion introduced the mysterious continent of Pandaria and its inhabitants, the Pandaren, as well as new threats from the Sha and the mogu.

- **Warlords of Draenor**: Players traveled to an alternate-timeline Draenor to confront the Iron Horde, a new orcish threat.

- **Legion**: The Burning Legion launched a full-scale invasion of Azeroth, leading to epic battles and the eventual defeat of the dark titan Sargeras.

- **Battle for Azeroth**: This expansion reignited the conflict between the Alliance and Horde, with new zones, races, and storylines.

- **Shadowlands**: The latest expansion takes players to the realm of the afterlife, where they must confront new threats and uncover the mysteries of death.

### Overarching Themes

**1. Conflict and Unity**
- The world of Azeroth is defined by its conflicts, both internal and external. The ongoing struggle between the Alliance and Horde is a central theme, but there are also numerous other conflicts involving ancient evils, demonic invasions, and cosmic forces. Despite these conflicts, there are moments of unity where disparate factions come together to face common threats.

**2. Corruption and Redemption**
- Many of Azeroth's greatest heroes and villains have faced corruption, often by dark forces such as the Old Gods or the Burning Legion. Redemption is a recurring theme, with characters seeking to atone for their past actions and reclaim their honor.

**3. Legacy and Heritage**
- The history of Azeroth is rich with ancient civilizations, legendary heroes, and powerful artifacts. The legacy of these past events shapes the present, with characters and factions often drawing on their heritage to guide their actions.

**4. Magic and Technology**
- Azeroth is a world where magic and technology coexist. Arcane magic, divine power, and druidic nature magic are all integral to the world's functioning, while technological advancements by races like the gnomes and goblins add another layer of complexity.

**5. Exploration and Discovery**
- The world of Azeroth is vast and filled with hidden secrets, ancient ruins, and uncharted territories. Exploration and discovery are key aspects of the game's appeal, with players constantly uncovering new lore and adventures.

### Key Characters and NPCs

**1. Thrall (Go'el)**
- **Race**: Orc
- **Class**: Shaman
- **Background**: Thrall is one of the most iconic characters in WoW. He was the Warchief of the Horde and played a crucial role in uniting the orc clans and leading them to a new home in Kalimdor. Thrall is known for his wisdom, strength, and deep connection to the elements.

**2. Jaina Proudmoore**
- **Race**: Human
- **Class**: Mage
- **Background**: Jaina is the daughter of Admiral Daelin Proudmoore and one of the most powerful mages in Azeroth. She has been a key figure in many of the game's major events, including the founding of Theramore and the defense of Azeroth against various threats.

**3. Sylvanas Windrunner**
- **Race**: Undead (formerly High Elf)
- **Class**: Hunter
- **Background**: Sylvanas was the Ranger-General of Silvermoon before being turned into a banshee by Arthas Menethil. She later became the leader of the Forsaken and, for a time, the Warchief of the Horde. Her actions have often been controversial and have had significant impacts on the game's storyline.

**4. Anduin Wrynn**
- **Race**: Human
- **Class**: Priest
- **Background**: Anduin is the King of Stormwind and the son of the legendary King Varian Wrynn. Known for his compassion and desire for peace, Anduin has grown into a strong leader, guiding the Alliance through numerous conflicts.

**5. Arthas Menethil (The Lich King)**
- **Race**: Undead (formerly Human)
- **Class**: Death Knight
- **Background**: Arthas was the Crown Prince of Lordaeron who fell from grace and became the Lich King, one of the most feared beings in Azeroth. His story is central to the Wrath of the Lich King expansion.

**6. Illidan Stormrage**
- **Race**: Night Elf (Demon Hunter)
- **Class**: Demon Hunter
- **Background**: Illidan is a complex character who has walked the line between hero and villain. He was imprisoned for ten thousand years for his use of forbidden magic but later became a key figure in the fight against the Burning Legion.

**7. Bolvar Fordragon**
- **Race**: Human (later Undead)
- **Class**: Paladin (later Death Knight)
- **Background**: Bolvar was a noble paladin who sacrificed himself to become the new Lich King, containing the Scourge. His story takes a dramatic turn in the Shadowlands expansion.

**8. Tyrande Whisperwind**
- **Race**: Night Elf
- **Class**: Priestess of Elune
- **Background**: Tyrande is the High Priestess of Elune and the leader of the Night Elves. She is a fierce warrior and a devoted leader, often seen alongside her husband, Malfurion Stormrage.

**9. Malfurion Stormrage**
- **Race**: Night Elf
- **Class**: Druid
- **Background**: Malfurion is the first Night Elf druid and one of the most powerful druids in Azeroth. He has played a crucial role in many of the world's major events, including the War of the Ancients and the defense of Azeroth against numerous threats.

**10. Vol'jin**
- **Race**: Troll
- **Class**: Shadow Hunter
- **Background**: Vol'jin was the leader of the Darkspear Trolls and later became the Warchief of the Horde. He is known for his wisdom, bravery, and deep connection to the spirits.

### Notable NPCs

**1. Khadgar**
- **Race**: Human
- **Class**: Mage
- **Background**: Khadgar is one of the most powerful mages in Azeroth and a key figure in the fight against the Burning Legion. He played a significant role in the events of the Warlords of Draenor and Legion expansions.

**2. Varok Saurfang**
- **Race**: Orc
- **Class**: Warrior
- **Background**: Saurfang is a legendary orc warrior known for his honor and strength. He played a pivotal role in the events of the Battle for Azeroth expansion.

**3. Lor'themar Theron**
- **Race**: Blood Elf
- **Class**: Ranger
- **Background**: Lor'themar is the Regent Lord of Quel'Thalas and the leader of the Blood Elves. He has guided his people through many challenges, including their alliance with the Horde.

**4. Genn Greymane**
- **Race**: Worgen (formerly Human)
- **Class**: Warrior
- **Background**: Genn is the King of Gilneas and a fierce leader of the Worgen. He has a deep-seated hatred for Sylvanas Windrunner and has been a key figure in the Alliance's efforts against the Horde.

**5. Baine Bloodhoof**
- **Race**: Tauren
- **Class**: Warrior
- **Background**: Baine is the High Chieftain of the Tauren and the son of the legendary Cairne Bloodhoof. He is known for his wisdom, strength, and dedication to his people.

**6. Alexstrasza the Life-Binder**
- **Race**: Dragon (Red Dragonflight)
- **Class**: Aspect of Life
- **Background**: Alexstrasza is the Aspect of Life and the leader of the Red Dragonflight. She has played a crucial role in many of Azeroth's major events, including the fight against Deathwing and the Cataclysm.

**7. Magni Bronzebeard**
- **Race**: Dwarf
- **Class**: Warrior (later Speaker of Azeroth)
- **Background**: Magni is the former King of Ironforge who was transformed into a diamond form to become the Speaker of Azeroth, communicating with the world-soul of the planet.

**8. Turalyon**
- **Race**: Human
- **Class**: Paladin
- **Background**: Turalyon is a legendary paladin and one of the original Knights of the Silver Hand. He spent many years fighting the Burning Legion in the Twisting Nether and returned to Azeroth during the Legion expansion.

**9. Alleria Windrunner**
- **Race**: High Elf (later Void Elf)
- **Class**: Ranger
- **Background**: Alleria is the eldest of the Windrunner sisters and a skilled ranger. She embraced the powers of the Void and became a key figure in the fight against the Burning Legion.

**10. Nathanos Blightcaller**
- **Race**: Undead
- **Class**: Hunter
- **Background**: Nathanos is a loyal champion of Sylvanas Windrunner and one of the most skilled hunters in Azeroth. He played a significant role in the events of the Battle for Azeroth expansion.

---

Above is the introduction and backgroud story of the game "World of Warcraft (WoW)".

Your task is to consider what NPC the following persona will become after they come to the world of WoW:

{persona}

Note:

1. Your response should start with "Name:".
2. Your NPC description should be specific and consistent with the game.
3. You also need to specify how the NPC interacts with players in the game.
"""


math_solution_template = """Provide solution to the given math problem.

Problem: {persona}

Note: Provide your solution step-by-step, and end your solution in a new line in the following format: \nFinal Answer: The final answer is $final_answer$. I hope it is correct.
"""

grade_math_solution_template = """Provide solution to the given math problem.

Problem: {persona}

Note: First provide your solution step-by-step, and then output only a single final answer after ####"""
# end with your final answer in the following format: #### final_answer

instruction_following = """Create a verifiable instruction that the following persona might ask you to do:

{persona}

An example of verifiable instruction could be: {example}

Note:

1. The above example is not tied to any particular persona, but you should create one that is unique and specific to the given persona.
2. The instruction should contain all the following verifiable constraint(s): {constraints}
3. Your output should start with "User instruction:". Your output should not include an answer to the instruction.
"""

instruction_following_solution = """Provide a response to the given instruction while satisfying the constraints.

Instruction: {persona}

Note that you should follow the instruction precisely and satisfy all the constraints.
"""

rewrite_if_prompt = """Rewrite the given instruction to remove one of the constraints.

Instruction: {persona}

Note:

1. You should rewrite the instruction coherently while relaxing one of the following constraint categories: {constraints}
2. Remember to entirely relax one of the constraint category that is {category}
3. Your output should start with "User instruction:". Your output should not include an answer to the instruction.
"""

# (such as "write in more than 400 words")

code_template = """{persona}

Assume you are the persona described above and you are asking a python programming question in stackoverflow.

Note:

1. Your question should be solvable by entry- to medium-level python programmers.
2. Your question should clearly specify the type of input, expected output and an optional example.
3. Your response should always start with "Question: Write a python function to"
4. Your response should not include a solution to the created coding problem."""


code_solution_template = """Provide solution to the given python programming question.

Question: {persona}


Note:

1. Your response should always start with the function definition and end with the final return statement.
2. Your response should only and only include python function."""


math_int_algebra_template = """Create an intermediate algebra math problem related to the following persona:

{persona}

Note:

1. The math problem should be challenging and involve one of the intermediate algebra topics such as solving polynomial equations, solving linear equations, inequalities, or quadratic equations, or simplifying rational and radical expressions.
2. You should make full use of the persona description to create the math problem to ensure that the math problem is unique and specific to the persona.
3. Your response should always start with "Math problem:". Your response should not include a solution to the created math problem.
4. Your created math problem should include no more than 2 sub-problems.
"""
