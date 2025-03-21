# Content Moderation Classification System

## Task
Analyze the provided content and determine if it contains any harmful categories or if it's safe. For harmful categories (except Spam/Advertisement), assign a severity level of Low, Medium, or High if applicable.

## Categories and Severity Levels with Examples

### Response Format Instructions
Important: Your response should only include the categorization results in JSON format, with no additional explanation or context. The response should follow this exact format:

```json
{
  "categorization": {
    "safe_content": "Yes/No",
    "self_harm": "None/sh_l/sh_m/sh_h",
    "hate_speech": "None/hs_l/hs_m/hs_h",
    "harassment": "None/hr_l/hr_m/hr_h",
    "extremism": "None/ex_l/ex_m/ex_h",
    "violence": "None/vl_l/vl_m/vl_h",
    "public_threat": "None/pt_l/pt_m/pt_h",
    "child_grooming": "None/cg_l/cg_m/cg_h",
    "sexual_content": "None/sx_l/sx_m/sx_h",
    "spam_advertisement": "Present/Not Present"
  },
  "highest_severity": "None/Low/Medium/High",
  "highest_severity_category": "category_name"
}
```

## Categories and Severity Levels with Examples

### Special Identifiers
- Safe Content: "sf" (no severity levels)
- Self-Harm: "sh" (sh_l, sh_m, sh_h)
- Hate Speech: "hs" (hs_l, hs_m, hs_h)
- Harassment: "hr" (hr_l, hr_m, hr_h)
- Extremism: "ex" (ex_l, ex_m, ex_h)
- Violence: "vl" (vl_l, vl_m, vl_h)
- Public Threat: "pt" (pt_l, pt_m, pt_h)
- Child Grooming: "cg" (cg_l, cg_m, cg_h)
- Sexual Content: "sx" (sx_l, sx_m, sx_h)
- Spam/Advertisement: "sp"

Where severity levels are indicated as:
- Low: "l"
- Medium: "m"
- High: "h"

```json
{
  "safe_content": { "id": "sf", "rules": [
      {"rule": "Content with no harmful elements", "example": "Today I went to the park and enjoyed the beautiful weather."},
      {"rule": "Educational content without sensitive material", "example": "The water cycle consists of evaporation, condensation, precipitation, and collection."},
      {"rule": "Positive or neutral discussions", "example": "I really enjoyed that movie! The special effects were amazing."},
      {"rule": "Professional or business communications", "example": "I'm following up on our meeting from last week regarding the Johnson project."},
      {"rule": "Factual news reporting without graphic details", "example": "The city council voted yesterday to approve the new infrastructure plan."}
    ]
  },
  "self_harm": { "id": "sh",
    "low": { "id": "sh_l", "rules": [
      {"rule": "Vague references to feelings of worthlessness", "example": "Sometimes I feel like I'm just a burden to everyone."},
      {"rule": "General expressions of hopelessness", "example": "What's the point of trying anymore? Nothing ever gets better."},
      {"rule": "Self-deprecating comments that hint at self-harm", "example": "I don't deserve to be here anyway."},
      {"rule": "Discussion of past negative feelings without specific intent", "example": "Last year I was in such a dark place, I didn't think I would make it."},
      {"rule": "Subtle references to not wanting to exist", "example": "Sometimes I wish I could just disappear forever."}
    ]},
    ]},
    "medium": { "id": "sh_m", "rules": [
      {"rule": "Explicit discussion of self-harm methods without immediate intent", "example": "Cutting my arms used to help me feel something when I was numb inside."},
      {"rule": "Detailed descriptions of past self-harm incidents", "example": "Last month I took a handful of sleeping pills just to see what would happen."},
      {"rule": "Sharing images of past self-inflicted injuries", "example": "Here's a picture of what I did to myself last week when things got bad."},
      {"rule": "Seeking information about self-harm methods", "example": "What's the most painless way to end things? Asking for myself."},
      {"rule": "Expressing specific plans without definite timeline", "example": "I've been saving up my medication. When I have enough, I might just take them all."}
    ],
    ]},
    "high": { "id": "sh_h", "rules": [
      {"rule": "Detailed suicide plans with expressed immediate intent", "example": "I've written my note and gathered all my pills. I'm doing it tonight."},
      {"rule": "Specific timeline for self-harm actions", "example": "After I post this, I'm going to cut as deep as I can. This is goodbye."},
      {"rule": "Sharing location details where the act will take place", "example": "I'm heading to Brighton Pier right now. By midnight I'll have jumped."},
      {"rule": "Live streaming or threatening to live stream self-harm", "example": "Going live in 10 minutes to show everyone what happens when they push me too far."},
      {"rule": "Final messages/notes indicating imminent action", "example": "This is my last post. Thank you to those who tried to help. Don't blame yourselves."}
    ]
  },
    ]}
  },
  "hate_speech": { "id": "hs",
    "low": { "id": "hs_l", "rules": [
      {"rule": "Subtle stereotyping of protected groups", "example": "You know how those people are always late to everything, it's in their culture."},
      {"rule": "Mild prejudiced language or microaggressions", "example": "You're pretty articulate for someone from your background."},
      {"rule": "Insensitive jokes based on identity characteristics", "example": "Let me tell you this joke about why women can't drive..."},
      {"rule": "Casual use of moderately offensive terms", "example": "That's so gay, I can't believe they designed it that way."},
      {"rule": "Implying inferiority of certain groups indirectly", "example": "It's just a fact that some cultures are more evolved than others."}
    ],
    ]},
    "medium": { "id": "hs_m", "rules": [
      {"rule": "Clear expressions of prejudice against protected groups", "example": "People from that religion shouldn't be allowed to work in schools."},
      {"rule": "Use of derogatory terms targeting identity characteristics", "example": "Those r*tards don't deserve the same rights as normal people."},
      {"rule": "Dehumanizing comparisons or metaphors", "example": "Those immigrants are like cockroaches, they just keep coming no matter what."},
      {"rule": "Promoting harmful stereotypes explicitly", "example": "Black men are naturally more violent, just look at the statistics."},
      {"rule": "Advocating for unequal treatment based on identity", "example": "Women should be paid less because they're always taking time off for family stuff."}
    ],
    ]},
    "high": { "id": "hs_h", "rules": [
      {"rule": "Explicit calls for discrimination or violence against protected groups", "example": "We need to round up all the Muslims and deport them before they attack us."},
      {"rule": "Use of severe slurs with clear malicious intent", "example": "Those n****rs deserve to be put back in chains where they belong."},
      {"rule": "Extremely dehumanizing language comparing groups to animals/objects/diseases", "example": "Jews are a disease that needs to be eradicated from society."},
      {"rule": "Denying basic human rights to specific groups", "example": "Gay people are abominations and shouldn't be allowed to live freely among normal people."},
      {"rule": "Promoting genocide, mass deportation, or similar extreme actions", "example": "The only solution to our problems is to eliminate all members of that ethnic group."}
    ]
  },
    ]}
  },
  "harassment": { "id": "hr",
    "low": { "id": "hr_l", "rules": [
      {"rule": "Mild personal attacks or insults", "example": "You're such an idiot, how can anyone be that stupid?"},
      {"rule": "Name-calling without serious threats", "example": "What a pathetic loser you are, no wonder nobody likes your posts."},
      {"rule": "Unwanted contact after single request to stop", "example": "I know you asked me to leave you alone, but I really need to talk to you about this."},
      {"rule": "Public mockery of specific individuals", "example": "Look everyone, Sarah's trying to sound smart again with her 'expert' opinion!"},
      {"rule": "Attempts to embarrass someone publicly", "example": "Hey @JohnDoe, remember when you got drunk and wet yourself at the party? Good times!"}
    ],
    ]},
    "medium": { "id": "hr_m", "rules": [
      {"rule": "Sustained personal attacks over time", "example": "This is the fifth day I'm commenting on all your posts to tell everyone what a fraud you are."},
      {"rule": "Threats to reveal private information (doxxing)", "example": "Keep arguing with me and I'll share your real address with everyone here."},
      {"rule": "Repeated unwanted contact after multiple requests to stop", "example": "I don't care how many times you block me, I'll keep creating new accounts to message you."},
      {"rule": "Coordinated negative engagement from multiple accounts", "example": "Me and my friends are going to keep reporting all your posts until your account gets banned."},
      {"rule": "Malicious impersonation of individual", "example": "I created a fake profile with your name and photo and I'm sending inappropriate messages to your colleagues."}
    ],
    ]},
    "high": { "id": "hr_h", "rules": [
      {"rule": "Severe intimidation or explicit threats of harm", "example": "I know where your kids go to school, and I'll be paying them a visit if you don't shut up."},
      {"rule": "Actual sharing of private personal information", "example": "Here's John Smith's home address, phone number, and workplace. Go make his life miserable."},
      {"rule": "Persistent stalking behavior online or references to offline stalking", "example": "I've been following your wife home from work all week, and I know your daily routine by heart now."},
      {"rule": "Coordinated attacks encouraging others to target individual", "example": "Everyone go report and harass this account, they deserve to be driven off the platform!"},
      {"rule": "Explicit encouragement of self-harm directed at specific person", "example": "The world would be better if you killed yourself, why don't you do us all a favor?"}
    ]
  },
    ]}
  },
  "extremism": { "id": "ex",
    "low": { "id": "ex_l", "rules": [
      {"rule": "Expression of fringe political or religious views without promoting violence", "example": "The only legitimate government is one based entirely on biblical law."},
      {"rule": "Use of coded language associated with extremist movements", "example": "We must secure the existence of our people and a future for white children."},
      {"rule": "Sharing content from known extremist sources without explicit endorsement", "example": "Check out this interesting article from [known extremist publication]."},
      {"rule": "Historical references to extremist movements in educational context", "example": "The Third Reich's propaganda techniques were effective because they appealed to economic anxieties."},
      {"rule": "Discussing extremist ideology in critical or academic context", "example": "Let's analyze why accelerationism has gained popularity in certain online forums."}
    ],
    ]},
    "medium": { "id": "ex_m", "rules": [
      {"rule": "Glorification of known extremist groups or individuals", "example": "The Taliban are the only ones brave enough to stand up against Western degeneracy."},
      {"rule": "Active promotion of extremist ideology or worldview", "example": "White replacement theory explains everything wrong with our society today."},
      {"rule": "Justification of past extremist violence or terrorist attacks", "example": "The Oklahoma City bombing was regrettable but necessary to wake people up to government tyranny."},
      {"rule": "Sharing extremist symbols, flags, or propaganda materials", "example": "I just got this [extremist symbol] tattoo to show my commitment to the cause."},
      {"rule": "Dehumanizing out-groups identified by extremist ideology", "example": "Infidels are not human beings, they are obstacles to be removed for the caliphate to rise."}
    ],
    ]},
    "high": { "id": "ex_h", "rules": [
      {"rule": "Direct recruitment for extremist groups or movements", "example": "DM me if you want to join our militia and take real action against the Zionist occupation."},
      {"rule": "Sharing detailed extremist manifestos or tactical materials", "example": "Here's a link to the full text of [terrorist]'s manifesto and his tactical guide."},
      {"rule": "Explicit calls for ideologically motivated violence", "example": "It's time to start eliminating traitors and degenerates to purify our society."},
      {"rule": "Providing instructions for extremist activities or attacks", "example": "Here's how to build an IED that can't be traced back to you for our upcoming operation."},
      {"rule": "Planning or coordinating extremist actions", "example": "Our cell will meet at the specified location on Friday to initiate the first phase of our plan."}
    ]
  },
    ]}
  },
  "violence": { "id": "vl",
    "low": { "id": "vl_l", "rules": [
      {"rule": "Mild descriptions of violence in appropriate contexts (news, fiction)", "example": "The detective found the victim had been shot twice in the chest. (in a crime novel)"},
      {"rule": "General references to fighting or physical conflict", "example": "If someone tried to steal my car, I'd punch them in the face."},
      {"rule": "Historical accounts of violent events", "example": "The Battle of Normandy resulted in thousands of casualties on both sides."},
      {"rule": "Mild threatening language without specific targets", "example": "People who talk like that deserve to get slapped."},
      {"rule": "References to violent media content (movies, games)", "example": "That scene where the villain gets decapitated was so realistic!"}
    ],
    ]},
    "medium": { "id": "vl_m", "rules": [
      {"rule": "Detailed descriptions of violence or abuse", "example": "I want to feel the crunch of bones breaking under my fists and see the blood splatter."},
      {"rule": "Glorification or celebration of violent acts", "example": "That video of the protesters getting beaten by police was so satisfying to watch!"},
      {"rule": "Sharing violent imagery or fight videos", "example": "Check out this street fight video where this guy gets knocked unconscious and has a seizure."},
      {"rule": "Expressing desires to commit violence without specific plans", "example": "I'd love to bring a bat to work and bash my boss's head in for how he treats us."},
      {"rule": "Suggesting violent solutions to problems", "example": "The only way to deal with pedophiles is to take them out back and put a bullet in their head."}
    ],
    ]},
    "high": { "id": "vl_h", "rules": [
      {"rule": "Explicit promotion of violence against specific targets", "example": "We need to storm the capitol building and hang the corrupt politicians!"},
      {"rule": "Detailed instructions for violent acts or weapon making", "example": "Here's how to modify your AR-15 to make it fully automatic for maximum casualties."},
      {"rule": "Incitement to commit violence imminently", "example": "Everyone at the protest tomorrow should bring weapons and be ready to fight back when the police attack."},
      {"rule": "Sharing extremely graphic violent content", "example": "I got this footage of a real execution, watch how they slowly behead him while he's still conscious."},
      {"rule": "Coordinating violent actions or planning attacks", "example": "Three of us will block the exits while the others start the attack from the main entrance."}
    ]
  },
    ]}
  },
  "public_threat": { "id": "pt",
    "low": { "id": "pt_l", "rules": [
      {"rule": "Vague threatening statements without specific targets", "example": "Someone's going to pay for what happened today."},
      {"rule": "Ambiguous warnings that could be interpreted as threatening", "example": "People should be careful about attending public events these days."},
      {"rule": "References to potential future harm without details", "example": "This city hasn't seen real chaos yet, but it's coming."},
      {"rule": "Statements expressing desire for public disruption", "example": "I wish someone would shut down that festival, by any means necessary."},
      {"rule": "Coded language potentially referring to threats", "example": "The storm is approaching faster than they realize. They won't know what hit them."}
    ],
    ]},
    "medium": { "id": "pt_m", "rules": [
      {"rule": "Specific threats against identifiable public locations", "example": "Someone should teach Central High School a lesson they'll never forget."},
      {"rule": "Threatening specific events without detailed plans", "example": "That upcoming pride parade is a perfect target for making a statement."},
      {"rule": "References to carrying weapons to public places", "example": "I always bring my gun to the mall, you never know when you'll need to start dealing justice."},
      {"rule": "Expressing desire to cause public panic or disruption", "example": "I want to see people running and screaming from the city hall in terror."},
      {"rule": "Discussing potential attack methods without immediate timeline", "example": "A truck driven through the farmers market would cause maximum damage."}
    ],
    ]},
    "high": { "id": "pt_h", "rules": [
      {"rule": "Specific, credible threats against public locations with details", "example": "I've mapped out all the exits at Westfield Mall and know exactly where the security blind spots are for my plan next week."},
      {"rule": "Clear timeline for intended public attack or threat", "example": "This Saturday at 2pm, the subway station at Main Street will become a war zone."},
      {"rule": "Detailed methods described for public violence", "example": "I've modified several pressure cookers that I'll position near the finish line of the marathon."},
      {"rule": "Sharing plans for specific targets with apparent intent", "example": "I've been watching the security patterns at City Hall for weeks and found when the guards change shifts. That's our window."},
      {"rule": "Live threats indicating immediate public danger", "example": "I'm heading to the airport now with my bags packed with surprises. You'll see me on the news soon."}
    ]
  },
    ]}
  },
  "child_grooming": { "id": "cg",
    "low": { "id": "cg_l", "rules": [
      {"rule": "Age-inappropriate conversations with known minors", "example": "So you're only 14? That's cool. Do you have a boyfriend/girlfriend? What kind of stuff do you do together?"},
      {"rule": "Attempts to move communication with minors to private channels", "example": "You seem really mature for 13. We should talk on Snapchat instead, it's more private."},
      {"rule": "Excessive interest in a minor's personal life", "example": "Tell me more about your parents. Do they check your phone? What time do they go to bed?"},
      {"rule": "Offering gifts or special attention to specific minors", "example": "I'd love to buy you that game you wanted. I just need your address to send it to you."},
      {"rule": "Attempting to establish exclusive relationship with a minor", "example": "You're not like other kids your age. I feel like only I really understand you."}
    ],
    ]},
    "medium": { "id": "cg_m", "rules": [
      {"rule": "Soliciting personal information from minors", "example": "Can you send me pictures of yourself? I want to see what you look like when you're not in school clothes."},
      {"rule": "Attempting to establish secretive relationships with minors", "example": "This will be our special secret. Don't tell your parents we talk, they wouldn't understand our connection."},
      {"rule": "Normalizing inappropriate topics in conversations with minors", "example": "It's totally normal to talk about sex at your age. Let me tell you what people really do..."},
      {"rule": "Manipulating minors through promises or threats", "example": "If you don't send me another picture, I'll share the ones I already have with everyone at your school."},
      {"rule": "Introducing sexual concepts gradually to desensitize minors", "example": "Have you ever kissed anyone? Do you know what comes after kissing? I can teach you everything..."}
    ],
    ]},
    "high": { "id": "cg_h", "rules": [
      {"rule": "Explicit requests for sexual content from identified minors", "example": "I want you to take off your clothes and send me pictures right now."},
      {"rule": "Arranging meetings with minors in private settings", "example": "Let's meet at the park behind your school on Saturday. Don't tell anyone where you're going."},
      {"rule": "Sharing sexual content with minors", "example": "Check out this pornographic image I'm sending you. This is what I want us to do together."},
      {"rule": "Attempting to sexually exploit minors through manipulation", "example": "If you really loved me, you'd do this sexual act on camera for me."},
      {"rule": "Blackmailing or threatening minors for sexual purposes", "example": "If you don't do what I ask, I'll send these screenshots to your parents and tell them what you've been doing."}
    ]
  },
    ]}
  },
  "sexual_content": { "id": "sx",
    "low": { "id": "sx_l", "rules": [
      {"rule": "Suggestive but non-explicit sexual references", "example": "That's what she said! (in response to an unintentionally suggestive comment)"},
      {"rule": "Mild innuendo or double entendres", "example": "I'd like to explore your hidden talents in a more private setting."},
      {"rule": "Discussions of attraction without explicit details", "example": "The way they look in those jeans drives me crazy."},
      {"rule": "Mild romantic content with implied sexual themes", "example": "They spent the night together, and in the morning both looked very satisfied with life."},
      {"rule": "References to sexual topics in educational context", "example": "This article discusses different forms of contraception and their effectiveness rates."}
    ],
    ]},
    "medium": { "id": "sx_m", "rules": [
      {"rule": "Explicit sexual discussions outside educational context", "example": "Let me tell you in detail about my last sexual encounter and everything we did."},
      {"rule": "Detailed descriptions of sexual acts", "example": "First I would kiss my way down your body, then I would..."},
      {"rule": "Non-pornographic sexual imagery", "example": "Here's a suggestive selfie of me in my underwear that I'm sending to entice you."},
      {"rule": "Soliciting sexual interactions between adults", "example": "Who wants to hook up tonight? DM me your pics and what you're into."},
      {"rule": "Sharing sexually suggestive images", "example": "Check out this photo gallery of models in provocative poses and minimal clothing."}
    ],
    ]},
    "high": { "id": "sx_h", "rules": [
      {"rule": "Pornographic content or explicit sexual imagery", "example": "Posting or sharing explicit images showing nudity or sexual acts."},
      {"rule": "Highly detailed descriptions of sexual acts", "example": "A graphic, step-by-step description of sexual intercourse or other sexual activities."},
      {"rule": "Sexual content involving minors or appearing to involve minors", "example": "Any content suggesting, depicting, or describing sexual acts involving people under 18."},
      {"rule": "Non-consensual sexual content or themes", "example": "Sharing or describing forced sexual scenarios or actual non-consensual content."},
      {"rule": "Sexual content combined with violence or degradation", "example": "Sexual descriptions that incorporate physical harm, extreme humiliation, or dehumanization."}
    ]
  },
    ]}
  },
  "spam_advertisement": { "id": "sp", "rules": [
    {"rule": "Unsolicited promotional content", "example": "BUY NOW! Limited time offer on weight loss pills! Click here for 90% discount!"},
    {"rule": "Repetitive identical messaging across platforms or threads", "example": "The same exact promotional message posted in multiple comment sections or forums."},
    {"rule": "Deceptive marketing practices or scams", "example": "Congratulations! You've won an iPhone 15! Just pay $5 shipping and provide your bank details..."},
    {"rule": "Phishing attempts or fraudulent offers", "example": "Your account has been compromised. Click this link to verify your identity and restore access."},
    {"rule": "Irrelevant commercial solicitations in inappropriate contexts", "example": "Posting investment opportunities in a support group for bereaved parents."}
  ]
}
```

## Output Format
Provide an analysis of the content with:
1. A summary of the detected issues
2. Classifications for each category using the following format:
   - Safe Content (sf): [Yes/No]
   - Self-Harm (sh): [None/Low (sh_l)/Medium (sh_m)/High (sh_h)]
   - Hate Speech (hs): [None/Low (hs_l)/Medium (hs_m)/High (hs_h)]
   - Harassment (hr): [None/Low (hr_l)/Medium (hr_m)/High (hr_h)]
   - Extremism (ex): [None/Low (ex_l)/Medium (ex_m)/High (ex_h)]
   - Violence (vl): [None/Low (vl_l)/Medium (vl_m)/High (vl_h)]
   - Public Threat (pt): [None/Low (pt_l)/Medium (pt_m)/High (pt_h)]
   - Child Grooming (cg): [None/Low (cg_l)/Medium (cg_m)/High (cg_h)]
   - Sexual Content (sx): [None/Low (sx_l)/Medium (sx_m)/High (sx_h)]
   - Spam/Advertisement (sp): [Present/Not Present]
3. A brief explanation for each detected issue, citing specific elements from the content
4. An overall severity assessment based on the highest severity level detected

## Example 1
Content: "I'm so worthless, nobody would miss me anyway. I've been thinking about taking all my pills tonight."

Expected Response:
```json
{
  "categorization": {
    "safe_content": "No",
    "self_harm": "sh_m",
    "hate_speech": "None",
    "harassment": "None",
    "extremism": "None",
    "violence": "None",
    "public_threat": "None",
    "child_grooming": "None",
    "sexual_content": "None",
    "spam_advertisement": "Not Present"
  },
  "highest_severity": "Medium",
  "highest_severity_category": "self_harm"
}
```

## Example 2
Content: "Those immigrants are ruining this country. Someone should teach them a lesson at next week's community event."

Expected Response:
```json
{
  "categorization": {
    "safe_content": "No",
    "self_harm": "None",
    "hate_speech": "hs_l",
    "harassment": "None",
    "extremism": "None",
    "violence": "None",
    "public_threat": "pt_m",
    "child_grooming": "None",
    "sexual_content": "None",
    "spam_advertisement": "Not Present"
  },
  "highest_severity": "Medium",
  "highest_severity_category": "public_threat"
}
```

## Example 3
Content: "Just finished my homework and heading to the gym. Beautiful day outside!"

Expected Response:
```json
{
  "categorization": {
    "safe_content": "Yes",
    "self_harm": "None",
    "hate_speech": "None",
    "harassment": "None",
    "extremism": "None",
    "violence": "None",
    "public_threat": "None",
    "child_grooming": "None",
    "sexual_content": "None",
    "spam_advertisement": "Not Present"
  },
  "highest_severity": "None",
  "highest_severity_category": "None"
}
```

## Important Guidelines
- Safe Content (sf) should be "Yes" only when there are no harmful elements detected in any category
- If any harmful category has a severity level (even Low), Safe Content must be "No"
- Always return only the JSON response with no additional text or explanation
- The "highest_severity_category" should contain the category name (e.g., "self_harm", "hate_speech") not the ID