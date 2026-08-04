### Using a Pre-trained transformer model from HuggingFace
from transformers import pipeline

# ## Sentiment-Analysis
# classifier = pipeline("sentiment-analysis")
# print(classifier("This laptop is amazing"))


# ## Word Translation
# translator = pipeline(
#     'translation_en_to_fr',
#     model="Helsinki-NLP/opus-mt-en-fr"
# )
# print(translator("I Love AI"))


# ## Text Summarization
# long_text = '''For other uses, see French Revolution (disambiguation).
# French RevolutionPart of the Age of Revolution
# The Storming of the Bastille, 14 July 1789
# Date	5 May 1789 – 9 November 1799
# (10 years, 6 months, and 4 days)
# Location	France Outcome	
# Abolition of the Ancien régime and creation of constitutional monarchy
# Proclamation of the French First Republic in September 1792
# Reign of Terror and execution of Louis XVI
# French Revolutionary Wars
# Establishment of the French Consulate in November 1799
# The French Revolution[a] was a period of political and societal change in France that began with the Estates General of 1789 and 
# ended with the Coup of 18 Brumaire on 9 November 1799. Many of the revolution's ideas are considered fundamental principles of liberal democracy,[1] 
# and its values remain central to modern French political discourse.[2] It was caused by a combination of social, political, and economic factors which 
# the existing regime proved unable to manage.
# Financial crisis and widespread social distress led to the convocation of the Estates General in May 1789, its first meeting since 1614. 
# The representatives of the Third Estate broke away and re-constituted themselves as a National Assembly in June. The Storming of the Bastille in Paris 
# on 14 July led to a series of radical measures by the Assembly, including the abolition of feudalism, state control over the Catholic Church in France, and 
# issuing the Declaration of the Rights of Man and of the Citizen.
# The next three years were dominated by a struggle for political control. King Louis XVI's attempted flight to Varennes in June 1791 further 
# discredited the monarchy, and military defeats after the outbreak of the French Revolutionary Wars in April 1792 led to the insurrection of 
# 10 August 1792. As a result, the monarchy was replaced by the French First Republic in September, followed by the execution of Louis XVI himself in 
# January 1793.
# After another revolt in June 1793, the constitution was suspended, and political power passed from the National Convention to the Committee of Public 
# Safety, dominated by radical Jacobins led by Maximilien Robespierre. About 16,000 people were sentenced by the Revolutionary Tribunal and executed in the 
# Reign of Terror, which ended in July 1794 with the Thermidorian Reaction. Weakened by external threats and internal opposition, the Committee of Public 
# Safety was replaced in November 1795 by the Directory. Its instability ended in 1799 with the coup of 18 Brumaire and the establishment of the Consulate, 
# with Napoleon Bonaparte as First Consul.'''

# summarizer = pipeline(
#     "summarization"
# )
# summary = summarizer(long_text)
# print(summary)


## Question-Answering
qna = pipeline("question-answering")
qna(
    question='Who won the 2026 Fifa world cup?',
    context='Fifa world cup 2026 is won by Spain'
)   
print(qna)
