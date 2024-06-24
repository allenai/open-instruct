from datasets import load_dataset
import json
import random

ds = load_dataset("TIGER-Lab/MATH-plus")

"{'socratic': 533383, 'mathstackexchange': 1484629, 'stackexchange': 317208}"

# [ { 
#     "role": "user", 
#     "content": "Question: Gdańsk (, ; German: \"\" , ) is a Polish city on the Baltic coast. It is the capital of the Pomeranian Voivodeship, Poland's principal seaport and is also the centre of the country's fourth-largest metropolitan area. \n\nThe city lies on the southern edge of Gdańsk Bay (of the Baltic Sea), in a conurbation with the city of Gdynia, spa town of Sopot, and suburban communities, which together form a metropolitan area called the Tricity (\"Trójmiasto\"), with a population approaching 1.4 million. Gdańsk itself has a population of 460,427 (December 2012), making it the largest city in the Pomerania region of Northern Poland. \n\nGdańsk is the capital of Gdańsk Pomerania and the largest city of Kashubia. With its origins as a Polish stronghold erected in the 980s by Mieszko I of Poland, the city's history is complex, with periods of Polish rule, periods of Prussian or German rule, and periods of autonomy or self-rule as a \"free city\". Between the world wars, the Free City of Danzig was in a customs union with Poland and was located between German East Prussia and the so-called Polish Corridor. \n\nGdańsk lies at the mouth of the Motława River, connected to the Leniwka, a branch in the delta of the nearby Vistula River, which drains 60 percent of Poland and connects Gdańsk with the Polish capital, Warsaw. Together with the nearby port of Gdynia, Gdańsk is also an important industrial center. In the late Middle Ages it was an important seaport and shipbuilding town, and in the 14th and 15th centuries a member of the Hanseatic League. 1. Which bay is Gdansk on?\n2. Which sea is that bay part of?\n3. Is Gdansk a capital?\n4. Of what?\n5. Who was the first ruler?\n6. Name one of the countries that has ruled it?\n7. And another?\n8. And one more?\n9. Has it ever been under self-rule?\n10. What was it called between the world wars?\n11. Is it near a river?\n12. Which one?\n13. Which is connected to ? Provide a numbered list of answers.\n****\nA numbered of answers: 1. Gdańsk Bay\n2. the Baltic Sea\n3. Yes\n4. Pomeranian Voivodeship\n5. Mieszko I\n6. Poland\n7. German\n8. Prussian\n9. Yes\n10. Danzig\n11. Yes\n12. the Motława River\n13. the Leniwka\nQ: Ferguson, Missouri (CNN) -- As the St. Louis suburb of Ferguson calmed Friday after nights of protests over the fatal shooting of an unarmed black teen, the question remains: Where's the police officer who pulled the trigger? \n\nOfficer Darren Wilson, 28, shot Michael Brown on August 9. The shooting sparked days of violent protests in Ferguson as residents demanded his arrest. \n\nSeparate federal and local investigations are under way, and Wilson -- who has received death threats -- has disappeared from public view. \n\nGovernor orders drawdown of National Guard in Ferguson \n\nHere's what is known about his whereabouts. \n\nWhere is he now? \n\nFew outside Wilson's family and authorities know for sure. \n\nWilson owns a house in a modest neighborhood about 20 miles from Ferguson. He bought the house shortly after he was divorced last year, according to the St. Louis Post-Dispatch. \n\nBut several neighbors have told CNN that Wilson left home before his name was released last week. \n\nWhat are his neighbors saying about his whereabouts? \n\nNot much. Most have shunned reporters' requests for interviews, and some put signs in their yards shooing away journalists. \n\n\"We don't know anything ... Pray for Peace,\" one read, according to the Post-Dispatch. \n\n\"We have 2 children. Do not knock!! No comment,\" another family wrote. \n\nAny trails on social media? \n\nThe newspaper reported that Wilson deactivated his social media accounts before his name went public. \n\nThe only social media presence for him now is from supporters, who have set up Facebook pages to support and raise money for him. \n\n1. unarmed black teen\n2. Darren Wilson shot Michael Brown\n3. Yes\n4. Few outside Wilson's family and authorities know for sure.\n5. about 20 miles from Ferguson\n6. No\n7. Most have shunned reporters' requests for interviews, and some put signs in their yards shooing away journalists\n8. federal and local investigations\n9. 28\n10. unknown\n11. The shooting sparked days of violent protests\n12. his arrest\n13. Most have not\n14. that Wilson left home\n15. shortly after he was divorced\n16. unknown\n17. Yes\n18. We have 2 children. Do not knock\n19. no\n20. Ferguson\n\nNumbered questions:\n1. Who was Michael Brown?\n2. What happened to him?\n3. Was Darren Wilson a cop?\n4. Where is Wilson now?\n5. Where does he live?\n6. Do his neighbors know where he is?\n7. How have they handled the publicity?\n8. What kind of ongoing investigations are there?\n9. How old is Wilson?\n10. Why did he shoot Michael Brown?\n11. How did local people react to this?\n12. What were they hoping to accomplish with the protests?\n13. Have any of his neighbors spoken to press?\n14. What did they tell CNN?\n15. When did Wilson buy the house?\n16. When was that?\n17. Did he receive death threats?\n18. What do the signs in the neighbors yards say?\n19. Is Wilson active online?\n20. What town was Brown shot in?\ninput: (CNN) -- At least one performer fell hard for Sunday night's Billboard Music Awards. \n\nNot sure what that means? Well, check out the top five moments from Sunday night's 2013 Billboard Music Awards: \n\n1. Miguel lands on a fan \n\nThe R & B singer accidentally landed on a woman in the mosh pit during a performance of his hit song \"Adorn.\" He was attempting a jump that went wrong. The fan appeared to be fine and the singer kept singing. Miguel later tweeted: \"got caught up in the moment, thank goodness Khyati is okay.\" \n\n2. Taylor Swift wins eight out of the 11 awards she was up for \n\nSwift is no stranger to taking to the stage to accept accolades, and on Sunday night she collected a few, including Billboard Artist of the Year. \n\n\"My album is kind of on the ends of the intense emotional spectrum,\" Swift said while accepting that award. \"You (fans) are the longest and best relationship I have ever had.\" \n\nShe also won Top Country Artist,Top Billboard 200 Artist, Top Female Artist, and Top Digital Songs Artist -- the last one a tie with singer Carly Rae Jepsen. Swift's album \"Red\" won in the Top Billboard 200 and Country Album categories and her single \"We Are Never Ever Getting Back Together\" collected the trophy for Top Country Song. \n\n3. Justin Bieber gets booed \n\nWhile accepting the first ever Milestone Award, the Biebs was both cheered and jeered. He appeared to reference the rough times he has had of late in his acceptance speech. 1. Who was the one performer who fell hard?\n2. was the fan ok?\n3. how many awards Taylor Swift won?\n4. Why did Justin get booed?\n5. Who was Carly Rae Jepsen tied with ?\n6. what song got top country song?\n7. Was the fan female or male that Miguel fell on?\n8. What did bieber talk about in his acceptance speech?\n9. what swift's ablum won the top billboard 200?\n10. Was \"Adorn\" a hit song? Return numbered answers in your output.\n\noutput:" 
#   }, { 
#     "role": "assistant",
#     "content": "1. Miguel\n2. Yes\n3. Eight\n4. unknown\n5. Swift\n6. \"We Are Never Ever Getting Back Together\"\n7. Female\n8. The rough times he has had of late.\n9. \"Red\"\n10. Yes" 
# } ]

dataset = load_dataset("allenai/tulu-v2-sft-mixture")
tulu_data = []
for elem in dataset["train"]:
    tulu_data.append(elem)
random.shuffle(tulu_data)

all_data = []
i = 0
for elem in ds["train"]:
    inst = {
        "dataset": elem["source"],
        "id": str(i),
        "messages": [
        {
            "role": "user",
            "content": elem["instruction"]
        },
        {
            "role": "assistant",
            "content": elem["output"]
        }
    ]}
    all_data.append(inst)
random.shuffle(all_data)

with open("/oe-adapt-default/jacobm/tulu-3-dev/data/metamath.jsonl", "w") as f_out:
    for elem in all_data:
        print(json.dumps(elem), file=f_out)

with open("/oe-adapt-default/jacobm/tulu-3-dev/data/metamath-100k.jsonl", "w") as f_out:
    for elem in all_data[:100000]:
        print(json.dumps(elem), file=f_out)

mixed_data = all_data + tulu_data
random.shuffle(mixed_data)
with open("/oe-adapt-default/jacobm/tulu-3-dev/data/metamath-with-tulu.jsonl", "w") as f_out:
    for elem in mixed_data:
        print(json.dumps(elem), file=f_out)

mixed_data_100k = all_data[:100000] + tulu_data
random.shuffle(mixed_data_100k)
with open("/oe-adapt-default/jacobm/tulu-3-dev/data/metamath-100k-with-tulu.jsonl", "w") as f_out:
    for elem in mixed_data_100k:
        print(json.dumps(elem), file=f_out)