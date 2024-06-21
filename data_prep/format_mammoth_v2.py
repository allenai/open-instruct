from datasets import load_dataset
import json
import random

ds = load_dataset("TIGER-Lab/WebInstructSub")

"{'socratic': 533383, 'mathstackexchange': 1484629, 'stackexchange': 317208}"

# [ { 
#     "role": "user", 
#     "content": "Question: Gdańsk (, ; German: \"\" , ) is a Polish city on the Baltic coast. It is the capital of the Pomeranian Voivodeship, Poland's principal seaport and is also the centre of the country's fourth-largest metropolitan area. \n\nThe city lies on the southern edge of Gdańsk Bay (of the Baltic Sea), in a conurbation with the city of Gdynia, spa town of Sopot, and suburban communities, which together form a metropolitan area called the Tricity (\"Trójmiasto\"), with a population approaching 1.4 million. Gdańsk itself has a population of 460,427 (December 2012), making it the largest city in the Pomerania region of Northern Poland. \n\nGdańsk is the capital of Gdańsk Pomerania and the largest city of Kashubia. With its origins as a Polish stronghold erected in the 980s by Mieszko I of Poland, the city's history is complex, with periods of Polish rule, periods of Prussian or German rule, and periods of autonomy or self-rule as a \"free city\". Between the world wars, the Free City of Danzig was in a customs union with Poland and was located between German East Prussia and the so-called Polish Corridor. \n\nGdańsk lies at the mouth of the Motława River, connected to the Leniwka, a branch in the delta of the nearby Vistula River, which drains 60 percent of Poland and connects Gdańsk with the Polish capital, Warsaw. Together with the nearby port of Gdynia, Gdańsk is also an important industrial center. In the late Middle Ages it was an important seaport and shipbuilding town, and in the 14th and 15th centuries a member of the Hanseatic League. 1. Which bay is Gdansk on?\n2. Which sea is that bay part of?\n3. Is Gdansk a capital?\n4. Of what?\n5. Who was the first ruler?\n6. Name one of the countries that has ruled it?\n7. And another?\n8. And one more?\n9. Has it ever been under self-rule?\n10. What was it called between the world wars?\n11. Is it near a river?\n12. Which one?\n13. Which is connected to ? Provide a numbered list of answers.\n****\nA numbered of answers: 1. Gdańsk Bay\n2. the Baltic Sea\n3. Yes\n4. Pomeranian Voivodeship\n5. Mieszko I\n6. Poland\n7. German\n8. Prussian\n9. Yes\n10. Danzig\n11. Yes\n12. the Motława River\n13. the Leniwka\nQ: Ferguson, Missouri (CNN) -- As the St. Louis suburb of Ferguson calmed Friday after nights of protests over the fatal shooting of an unarmed black teen, the question remains: Where's the police officer who pulled the trigger? \n\nOfficer Darren Wilson, 28, shot Michael Brown on August 9. The shooting sparked days of violent protests in Ferguson as residents demanded his arrest. \n\nSeparate federal and local investigations are under way, and Wilson -- who has received death threats -- has disappeared from public view. \n\nGovernor orders drawdown of National Guard in Ferguson \n\nHere's what is known about his whereabouts. \n\nWhere is he now? \n\nFew outside Wilson's family and authorities know for sure. \n\nWilson owns a house in a modest neighborhood about 20 miles from Ferguson. He bought the house shortly after he was divorced last year, according to the St. Louis Post-Dispatch. \n\nBut several neighbors have told CNN that Wilson left home before his name was released last week. \n\nWhat are his neighbors saying about his whereabouts? \n\nNot much. Most have shunned reporters' requests for interviews, and some put signs in their yards shooing away journalists. \n\n\"We don't know anything ... Pray for Peace,\" one read, according to the Post-Dispatch. \n\n\"We have 2 children. Do not knock!! No comment,\" another family wrote. \n\nAny trails on social media? \n\nThe newspaper reported that Wilson deactivated his social media accounts before his name went public. \n\nThe only social media presence for him now is from supporters, who have set up Facebook pages to support and raise money for him. \n\n1. unarmed black teen\n2. Darren Wilson shot Michael Brown\n3. Yes\n4. Few outside Wilson's family and authorities know for sure.\n5. about 20 miles from Ferguson\n6. No\n7. Most have shunned reporters' requests for interviews, and some put signs in their yards shooing away journalists\n8. federal and local investigations\n9. 28\n10. unknown\n11. The shooting sparked days of violent protests\n12. his arrest\n13. Most have not\n14. that Wilson left home\n15. shortly after he was divorced\n16. unknown\n17. Yes\n18. We have 2 children. Do not knock\n19. no\n20. Ferguson\n\nNumbered questions:\n1. Who was Michael Brown?\n2. What happened to him?\n3. Was Darren Wilson a cop?\n4. Where is Wilson now?\n5. Where does he live?\n6. Do his neighbors know where he is?\n7. How have they handled the publicity?\n8. What kind of ongoing investigations are there?\n9. How old is Wilson?\n10. Why did he shoot Michael Brown?\n11. How did local people react to this?\n12. What were they hoping to accomplish with the protests?\n13. Have any of his neighbors spoken to press?\n14. What did they tell CNN?\n15. When did Wilson buy the house?\n16. When was that?\n17. Did he receive death threats?\n18. What do the signs in the neighbors yards say?\n19. Is Wilson active online?\n20. What town was Brown shot in?\ninput: (CNN) -- At least one performer fell hard for Sunday night's Billboard Music Awards. \n\nNot sure what that means? Well, check out the top five moments from Sunday night's 2013 Billboard Music Awards: \n\n1. Miguel lands on a fan \n\nThe R & B singer accidentally landed on a woman in the mosh pit during a performance of his hit song \"Adorn.\" He was attempting a jump that went wrong. The fan appeared to be fine and the singer kept singing. Miguel later tweeted: \"got caught up in the moment, thank goodness Khyati is okay.\" \n\n2. Taylor Swift wins eight out of the 11 awards she was up for \n\nSwift is no stranger to taking to the stage to accept accolades, and on Sunday night she collected a few, including Billboard Artist of the Year. \n\n\"My album is kind of on the ends of the intense emotional spectrum,\" Swift said while accepting that award. \"You (fans) are the longest and best relationship I have ever had.\" \n\nShe also won Top Country Artist,Top Billboard 200 Artist, Top Female Artist, and Top Digital Songs Artist -- the last one a tie with singer Carly Rae Jepsen. Swift's album \"Red\" won in the Top Billboard 200 and Country Album categories and her single \"We Are Never Ever Getting Back Together\" collected the trophy for Top Country Song. \n\n3. Justin Bieber gets booed \n\nWhile accepting the first ever Milestone Award, the Biebs was both cheered and jeered. He appeared to reference the rough times he has had of late in his acceptance speech. 1. Who was the one performer who fell hard?\n2. was the fan ok?\n3. how many awards Taylor Swift won?\n4. Why did Justin get booed?\n5. Who was Carly Rae Jepsen tied with ?\n6. what song got top country song?\n7. Was the fan female or male that Miguel fell on?\n8. What did bieber talk about in his acceptance speech?\n9. what swift's ablum won the top billboard 200?\n10. Was \"Adorn\" a hit song? Return numbered answers in your output.\n\noutput:" 
#   }, { 
#     "role": "assistant",
#     "content": "1. Miguel\n2. Yes\n3. Eight\n4. unknown\n5. Swift\n6. \"We Are Never Ever Getting Back Together\"\n7. Female\n8. The rough times he has had of late.\n9. \"Red\"\n10. Yes" 
# } ]

{
    "dataset": "sharegpt", 
    "id": "sharegpt_Laiu9nC_0", 
    "messages": [{"role": "user", "content": "\u4e0a\u73ed\u901a\u52e4\u56db\u5c0f\u65f6\u662f\u4ec0\u4e48\u4f53\u9a8c\uff1f"}, {"role": "assistant", "content": "\u4e0a\u73ed\u901a\u52e4\u56db\u5c0f\u65f6\u7684\u4f53\u9a8c\u53ef\u80fd\u4f1a\u975e\u5e38\u75b2\u60eb\u548c\u6311\u6218\u6027\u3002\u8fd9\u79cd\u60c5\u51b5\u4e0b\uff0c\u60a8\u9700\u8981\u5728\u65e9\u6668\u8d77\u5e8a\u5f97\u66f4\u65e9\uff0c\u5e76\u5728\u4e0b\u73ed\u540e\u518d\u665a\u56de\u5bb6\u3002\u7531\u4e8e\u957f\u65f6\u95f4\u7684\u901a\u52e4\uff0c\u60a8\u53ef\u80fd\u4f1a\u611f\u5230\u8eab\u4f53\u75b2\u60eb\u548c\u7cbe\u795e\u75b2\u60eb\uff0c\u7279\u522b\u662f\u5982\u679c\u60a8\u9700\u8981\u5728\u62e5\u6324\u7684\u516c\u5171\u4ea4\u901a\u5de5\u5177\u4e0a\u5ea6\u8fc7\u5927\u91cf\u65f6\u95f4\u3002\u6b64\u5916\uff0c\u60a8\u7684\u5de5\u4f5c\u6548\u7387\u53ef\u80fd\u4f1a\u53d7\u5230\u5f71\u54cd\uff0c\u56e0\u4e3a\u60a8\u9700\u8981\u5728\u8def\u4e0a\u82b1\u8d39\u5f88\u591a\u65f6\u95f4\uff0c\u800c\u4e0d\u662f\u5728\u5de5\u4f5c\u4e2d\u3002\u8fd9\u53ef\u80fd\u4f1a\u5bfc\u81f4\u66f4\u591a\u7684\u538b\u529b\u548c\u7126\u8651\uff0c\u56e0\u4e3a\u60a8\u5fc5\u987b\u66f4\u597d\u5730\u7ba1\u7406\u60a8\u7684\u65f6\u95f4\u548c\u5de5\u4f5c\u4efb\u52a1\u3002\u5982\u679c\u60a8\u9762\u4e34\u8fd9\u6837\u7684\u60c5\u51b5\uff0c\u5efa\u8bae\u60a8\u8003\u8651\u8c03\u6574\u5de5\u4f5c\u65f6\u95f4\u6216\u5730\u70b9\uff0c\u6216\u8005\u5bfb\u627e\u66f4\u9760\u8fd1\u5de5\u4f5c\u5730\u70b9\u7684\u4f4f\u6240\uff0c\u4ee5\u51cf\u8f7b\u901a\u52e4\u7684\u538b\u529b\u3002"}]
}

all_data = []
socratic = []
mathstackexchange = []
stackexchange = []

i = 0
for elem in ds["train"]:
    inst = {
        "dataset": elem["source"],
        "id": str(i),
        "messages": [
        {
            "role": "user",
            "content": elem["question"]
        },
        {
            "role": "assistant",
            "content": elem["answer"]
        }
    ]}
    all_data.append(inst)
    if elem["source"] == "socratic":
        socratic.append(inst)
    elif elem["source"] == "mathstackexchange":
        mathstackexchange.append(inst)
    elif elem["source"] == "stackexchange":
        stackexchange.append(inst)
    else:
        print(f"unknown source: {elem['source']}")
    i += 1

random.shuffle(all_data)
random.shuffle(socratic)
random.shuffle(mathstackexchange)
random.shuffle(stackexchange)
with open("/oe-adapt-default/jacobm/tulu-3-dev/data/mammoth_v2-all.jsonl", "w") as f_out:
    for elem in all_data:
        print(json.dumps(elem), file=f_out)

with open("/oe-adapt-default/jacobm/tulu-3-dev/data/mammoth_v2-socratic.jsonl", "w") as f_out:
    for elem in socratic:
        print(json.dumps(elem), file=f_out)

with open("/oe-adapt-default/jacobm/tulu-3-dev/data/mammoth_v2-mathstackexchange.jsonl", "w") as f_out:
    for elem in mathstackexchange:
        print(json.dumps(elem), file=f_out)

with open("/oe-adapt-default/jacobm/tulu-3-dev/data/mammoth_v2-stackexchange.jsonl", "w") as f_out:
    for elem in stackexchange:
        print(json.dumps(elem), file=f_out)