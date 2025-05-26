# Prosperity-Lumon  
</details>
<details>
<summary><h2> Resources</h2></summary>

## Trading Terms Wiki 
[Trading Glossary](https://imc-prosperity.notion.site/Trading-glossary-19ee8453a09381478d7ce4e322dcaff4)  

## Algo Writing Wiki  
[Writing an Algorithm in Python](https://imc-prosperity.notion.site/Writing-an-Algorithm-in-Python-19ee8453a0938114a15eca1124bf28a1#19ee8453a09381638ed3cbd18cc93f28)  

## Previous Manual 
[Manual Git](https://github.com/gabsens/IMC-Prosperity-2-Manual/tree/master)

## Top Placing Git 
- [Second Place](https://github.com/ericcccsliu/imc-prosperity-2/blob/main/README.md)

- [Ninth Place](https://github.com/jmerle/imc-prosperity-2)

## Open-source Tools  
- [IMC Prosperity 3 Submitter](https://github.com/jmerle/imc-prosperity-3-submitter)  
- [IMC Prosperity 3 Backtester](https://github.com/jmerle/imc-prosperity-3-backtester)  
- [IMC Prosperity 3 Visualizer](https://github.com/jmerle/imc-prosperity-3-visualizer)  
</details>
<details>
<summary><h2> Set Up</h2></summary>

## Create Environment 
- Install Conda
- Create environment 
```
conda create --name lumon python=3.12
```
- Activate environment 
```
conda activate lumon
```
- Install dependencies 
```
pip install -r requirements.txt 
```
## Run Code

Use this command to run
```console
prosperity3bt tutorial/tutorial.py 0 --vis --match-trades worse
```
- ```prosperity3bt``` is the backtester command
- ```tutorial/tutorial.py``` path to trader file
- ```0``` is the day to run the backtester for
- ```--vis``` is to open results in visualiser 
- ```--match-trades``` worse (need to test which option is most accurate but from reading chat sounds like this is it - matches your orders against trades that happened at worse prices than what you put out)

## Stay updated

run
```
pip install -U prosperity3bt
```
use the right round
```
prosperity3bt tutorial/tutorial.py 1 --vis --match-trades worse
```
# Final Result 
<details>
<summary><h2>Final Result</h2></summary>

- Country: 52  
- Overall: 180  
- Manual: 119

</details>



