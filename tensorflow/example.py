from tensorflow import seq2seq_transformer

if __name__ == "__main__":
    # training data
    # a top list of misspelled words provided this whoop
    pairs = [
        ("adress", "address"),
        ("accomodate", "accommodate"),
        ("acheive", "achieve"),
        ("arguement", "argument"),
        ("beleive", "believe"),
        ("begining", "beginning"),
        ("buisness", "business"),
        ("calender", "calendar"),
        ("cemetary", "cemetery"),
        ("changable", "changeable"),
        ("comming", "coming"),
        ("concious", "conscious"),
        ("consensus", "consensus"),
        ("definately", "definite"),
        ("disapoint", "disappoint"),
        ("embarrasment", "embarrassment"),
        ("enviroment", "environment"),
        ("existance", "existence"),
        ("exhilerate", "exhilarate"),
        ("foward", "forward"),
        ("friend", "friend"),
        ("goverment", "government"),
        ("grammer", "grammar"),
        ("gratefull", "grateful"),
        ("guarentee", "guarantee"),
        ("hieght", "height"),
        ("humerous", "humorous"),
        ("ignorence", "ignorance"),
        ("immediatly", "immediately"),
        ("independant", "independent"),
        ("inteligence", "intelligence"),
        ("jealosy", "jealousy"),
        ("jewlry", "jewelry"),
        ("judgement", "judgment"),
        ("knowlege", "knowledge"),
        ("libary", "library"),
        ("maintainence", "maintenance"),
        ("mathematicas", "mathematics"),
        ("mischievious", "mischievous"),
        ("neccessary", "necessary"),
        ("occurence", "occurrence"),
        ("ommision", "omission"),
        ("perserverance", "perseverance"),
        ("preceed", "precede"),
        ("priviledge", "privilege"),
        ("recieve", "receive"),
        ("recomend", "recommend"),
        ("refference", "reference"),
        ("religous", "religious"),
        ("seperate", "separate"),
        ("suprise", "surprise"),
        ("thourough", "thorough"),
        ("truely", "truly"),
        ("untill", "until"),
        ("wierd", "weird"),
        ("wich", "which"),
        ("writting", "writing")
    ]

    # initialize model
    s2s = seq2seq_transformer.Seq2SeqTransformer()

    # train model
    s2s.fit(pairs)

    # testing
    print(s2s.correct("worrysome"))
    print(s2s.correct("botherr"))
 