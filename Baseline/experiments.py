import language_tool_python

tool = language_tool_python.LanguageTool("en-US")

text = """Let the bird of loudest lay On the sole Arabian tree Herald sad and trumpet be, To whose sound chaste wings obey. But thou shrieking harbinger, Foul precurrer of the fiend, Augur of the fever's end, To this troop come thou not near. From this session interdict Every fowl of tyrant wing, Save the eagle, feather'd king; Keep the obsequy so strict. Let the priest in surplice white, That defunctive music can, Be the death-divining swan, Lest the requiem lack his right. And thou treble-dated crow, That thy sable gender mak'st With the breath thou giv'st and tak'st, 'Mongst our mourners shalt thou go. Here the anthem doth commence: Love and constancy is dead; Phoenix and the Turtle fled In a mutual flame from hence. So they lov'd, as Love in twain Had the essence but in one; Two distincts, division none: Number there in Love was slain. Hearts remote, yet not asunder; Distance and no space was seen 'Twixt this Turtle and his queen: But in them it were a wonder. So between them Love did shine That the Turtle saw his right Flaming in the Phoenix' sight: Either was the other's mine. Property was thus appalled That the self was not the same; Single nature's double name Neither two nor one was called. Reason, in itself confounded,"""
text = "Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love Love."
text = "To My Fairy Fancies NAY,  \
        Here she comes with charm.  \
        Come, go, fair one, and kiss me,  \
        And make me wondrous all the day.I never understood what was happening to me  \
        You walked out of the corner and then you walked back in  \
        You gave me hope and I cried a moment later  \
        You left me here in that empty corner all by myself  \
        But, you gave"
# get the matches
matches = tool.check(text)
# for match in matches:
#     print(match.ruleId)
for i in matches:
    print(i)
print(len(matches))
print(len(text.split()))
# print(matches)