# Prosperity-Lumon  

## Trading Terms Wiki  
[Trading Glossary](https://imc-prosperity.notion.site/Trading-glossary-19ee8453a09381478d7ce4e322dcaff4)  

## Algo Writing Wiki  
[Writing an Algorithm in Python](https://imc-prosperity.notion.site/Writing-an-Algorithm-in-Python-19ee8453a0938114a15eca1124bf28a1#19ee8453a09381638ed3cbd18cc93f28)  

## Open-source Tools  
- [IMC Prosperity 3 Submitter](https://github.com/jmerle/imc-prosperity-3-submitter)  
- [IMC Prosperity 3 Backtester](https://github.com/jmerle/imc-prosperity-3-backtester)  
- [IMC Prosperity 3 Visualizer](https://github.com/jmerle/imc-prosperity-3-visualizer)  

## Contributing

Use the following commands to make a python 3.12 enviornment and install requirements.txt (make sure you have conda)
```
conda create --name lumon python=3.12
conda activate lumon
pip install -r requirements.txt   
```

Use this command to run
```console
prosperity3bt tutorial/example-program.py 0 --vis --match-trades worse
```
1. prosperity3bt is the backtester command
2. tutorial/example-program.py path to trader file
3. 0 is the day to run the backtester for
4. --vis is to open results in visualiser 
5. --match-trades worse (need to test which option is most accurate but from reading chat sounds like this is it - matches your orders against trades that happened at worse prices than what you put out)