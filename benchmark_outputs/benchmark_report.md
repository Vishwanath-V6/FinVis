# Tata Motors FY25 FinVis Benchmark

- Questions: 128 (104 answerable, 24 unanswerable)
- Chunks: 2638 (1507 table chunks)

| System | Hit@5 | MRR | Answer correctness | Citation accuracy | Abstention accuracy | Median CPU ms |
|---|---:|---:|---:|---:|---:|---:|
| minilm | 79/104 (75.96%) | 0.6955 | 43/104 (41.35%) | 76/104 (73.08%) | 23/24 (95.83%) | 31.25 |
| tfidf | 92/104 (88.46%) | 0.6309 | 57/104 (54.81%) | 89/104 (85.58%) | 23/24 (95.83%) | 15.62 |

## Error analysis

### minilm

- retrieval_miss: 25
- incorrect_answer: 61
- wrong_citation: 0
- abstention_failure: 50

- `TATA001`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA002`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA003`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA004`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA007`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA008`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA009`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA013`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA014`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA015`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA016`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA017`: incorrect_answer; returned answer: '10,789 ₹ crore'
- `TATA018`: incorrect_answer; returned answer: '8,419 ₹ crore'
- `TATA019`: incorrect_answer; returned answer: '67,380 ₹ crore'
- `TATA020`: incorrect_answer; returned answer: '67,581 ₹ crore'
- `TATA027`: retrieval_miss, incorrect_answer; returned answer: '7,352 ₹ crore'
- `TATA037`: incorrect_answer; returned answer: '(922) ₹ crore'
- `TATA039`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA040`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA041`: retrieval_miss, incorrect_answer; returned answer: '33,780 ₹ crore'
- `TATA042`: retrieval_miss, incorrect_answer; returned answer: '28,429 ₹ crore'
- `TATA044`: incorrect_answer; returned answer: '(2,809) ₹ crore'
- `TATA052`: incorrect_answer; returned answer: '(51) ₹ crore'
- `TATA054`: incorrect_answer; returned answer: '(4,024) ₹ crore'
- `TATA060`: incorrect_answer; returned answer: '(646) ₹ crore'
- `TATA069`: incorrect_answer, abstention_failure; returned answer: None
- `TATA070`: incorrect_answer, abstention_failure; returned answer: None
- `TATA071`: incorrect_answer, abstention_failure; returned answer: None
- `TATA072`: incorrect_answer, abstention_failure; returned answer: None
- `TATA073`: incorrect_answer, abstention_failure; returned answer: None
- `TATA074`: incorrect_answer, abstention_failure; returned answer: None
- `TATA075`: incorrect_answer, abstention_failure; returned answer: None
- `TATA076`: incorrect_answer, abstention_failure; returned answer: None
- `TATA077`: incorrect_answer, abstention_failure; returned answer: None
- `TATA078`: incorrect_answer, abstention_failure; returned answer: None
- `TATA079`: incorrect_answer, abstention_failure; returned answer: None
- `TATA080`: incorrect_answer, abstention_failure; returned answer: None
- `TATA081`: incorrect_answer, abstention_failure; returned answer: None
- `TATA082`: incorrect_answer, abstention_failure; returned answer: None
- `TATA083`: incorrect_answer, abstention_failure; returned answer: None
- `TATA084`: incorrect_answer, abstention_failure; returned answer: None
- `TATA085`: incorrect_answer, abstention_failure; returned answer: None
- `TATA086`: incorrect_answer, abstention_failure; returned answer: None
- `TATA087`: incorrect_answer, abstention_failure; returned answer: None
- `TATA088`: incorrect_answer, abstention_failure; returned answer: None
- `TATA089`: incorrect_answer, abstention_failure; returned answer: None
- `TATA090`: incorrect_answer, abstention_failure; returned answer: None
- `TATA091`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA092`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA093`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA094`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA095`: incorrect_answer, abstention_failure; returned answer: None
- `TATA096`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA097`: incorrect_answer, abstention_failure; returned answer: None
- `TATA098`: incorrect_answer, abstention_failure; returned answer: None
- `TATA099`: incorrect_answer, abstention_failure; returned answer: None
- `TATA100`: incorrect_answer, abstention_failure; returned answer: None
- `TATA101`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA102`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA103`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA104`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA120`: abstention_failure; returned answer: '65,848 ₹ crore'
### tfidf

- retrieval_miss: 12
- incorrect_answer: 47
- wrong_citation: 0
- abstention_failure: 39

- `TATA003`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA007`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA017`: incorrect_answer, abstention_failure; returned answer: None
- `TATA018`: incorrect_answer, abstention_failure; returned answer: None
- `TATA019`: incorrect_answer, abstention_failure; returned answer: None
- `TATA020`: incorrect_answer, abstention_failure; returned answer: None
- `TATA037`: incorrect_answer; returned answer: '(922) ₹ crore'
- `TATA044`: incorrect_answer; returned answer: '(2,809) ₹ crore'
- `TATA052`: incorrect_answer; returned answer: '(51) ₹ crore'
- `TATA054`: incorrect_answer; returned answer: '(4,024) ₹ crore'
- `TATA060`: incorrect_answer; returned answer: '(646) ₹ crore'
- `TATA069`: incorrect_answer, abstention_failure; returned answer: None
- `TATA070`: incorrect_answer, abstention_failure; returned answer: None
- `TATA071`: incorrect_answer, abstention_failure; returned answer: None
- `TATA072`: incorrect_answer, abstention_failure; returned answer: None
- `TATA073`: incorrect_answer, abstention_failure; returned answer: None
- `TATA074`: incorrect_answer, abstention_failure; returned answer: None
- `TATA075`: incorrect_answer, abstention_failure; returned answer: None
- `TATA076`: incorrect_answer, abstention_failure; returned answer: None
- `TATA077`: incorrect_answer, abstention_failure; returned answer: None
- `TATA078`: incorrect_answer, abstention_failure; returned answer: None
- `TATA079`: incorrect_answer, abstention_failure; returned answer: None
- `TATA080`: incorrect_answer, abstention_failure; returned answer: None
- `TATA081`: incorrect_answer, abstention_failure; returned answer: None
- `TATA082`: incorrect_answer, abstention_failure; returned answer: None
- `TATA083`: incorrect_answer, abstention_failure; returned answer: None
- `TATA084`: incorrect_answer, abstention_failure; returned answer: None
- `TATA085`: incorrect_answer, abstention_failure; returned answer: None
- `TATA086`: incorrect_answer, abstention_failure; returned answer: None
- `TATA087`: incorrect_answer, abstention_failure; returned answer: None
- `TATA088`: incorrect_answer, abstention_failure; returned answer: None
- `TATA089`: incorrect_answer, abstention_failure; returned answer: None
- `TATA090`: incorrect_answer, abstention_failure; returned answer: None
- `TATA091`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA092`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA093`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA094`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA095`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA096`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA097`: incorrect_answer; returned answer: '439,695 ₹ crore'
- `TATA098`: incorrect_answer; returned answer: '434,016 ₹ crore'
- `TATA099`: incorrect_answer; returned answer: '373,847 ₹ crore'
- `TATA100`: incorrect_answer; returned answer: '365,185 ₹ crore'
- `TATA101`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA102`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA103`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA104`: retrieval_miss, incorrect_answer, abstention_failure; returned answer: None
- `TATA120`: abstention_failure; returned answer: '65,848 ₹ crore'
