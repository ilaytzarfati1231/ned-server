# Install

Both `ned-server` and `ned-client` need to be installed for the system to function correctly.

## Repositories:
- [`ned-server`](https://github.com/ilaytzarfati1231/ned-server)
- [`ned-client`](https://github.com/ilaytzarfati1231/ned-client)

## In `ned-server` run:
First Time ONLY:
```sh
python -m venv venv
```

To activate environment:
```sh
./venv/Scripts/activate
```

Install dependencies:
```sh
pip install -r requirements.txt
```

Run the server:
```sh
python app.py
```

## In `ned-client` run:
Install dependencies:
```sh
npm i
```

Run the client:
```sh
npm run dev
```
