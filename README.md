<h1>Primeclue</h1>
<h2>Data Mining / Artificial Intelligence</h2>

<h3>What does it do?</h3>
Primeclue is a data mining software / library. It reads numerical data, pushes it through mathematical functions
and yields classification.

<h3>How does it work?</h3>
As always, best place to look is the code itself, but here is a high level overview:<br>
Most important part of Primeclue are <i>trees</i> (which formally are <i>graphs</i>, but really simple ones). For example:

          data[1]  data[2]    pi     e       data[3]
               \       /      /       \       /
                \     /      /         \     /
                 \   /      /           \   /
                  max      /             \ /
                   \      /           multiply
                    \    /             /
                     subtract         /
                        \            /
                         \        sq root
                          \       /
                           \     /
                             add
                              |
                         Final result
                         
A tree describes how data flows through math functions to give final value. This value is later interpreted as either ```true``` or ```false```.
Execution of a tree starts at top nodes 
(which yield values from input data or math constants) and flows down to final node.
Primeclue creates lots of random trees and executes them on given data. 
Multiclass classification is achieved by using multiple trees, each classifying one class.

During training bad trees get removed and more random
trees is created to fill the space.
The best tree is executed against the testing set, and the result is displayed to a user. 
Obviously, the result on the testing set does not leak into training logic in any way. 

Actual implementation is slightly more complicated. Source code is your best friend here. 
Also, trees usually are a lot bigger than this example. This enables Primeclue to perform well on problems like
predicting companies bankruptcies and banknotes verification.

<h3>Can it actually learn anything?</h3>
Yes. Please refer to <a href="test_data/README.md">test data</a> to see results.

<h3>How to run it?</h3>

<h4>Docker image</h4>
Run Primeclue's image from docker:

```shell script
docker run -it -p 8080:80 -p 8180:8180 lukaszwojtow/primeclue
```

This will expose ports 8080 and 8180 on your machine such that browser can connect to them. Then go to:
http://localhost:8080/

By default, Primeclue saves data in user's home directory. It happens to be
`/root/Primeclue` inside the container, so it disappears between runs. Use docker volumes to save it permanently:
```shell script
docker run -it -p 8080:80 -p 8180:8180 -v /some/directory/here:/root/Primeclue lukaszwojtow/primeclue
```

 
<h4>Run from this repo</h4>
You need <b>cargo</b> and <b>npm</b> on your system. 
Run backend with command:

```shell script
cargo run --release
```

By default, it binds to `localhost:8180`. Port is set in <a href="backend/primeclue-api/src/rest.rs">primeclue-api/src/rest.rs</a> if you need to change it.<br>
To serve frontend to your browser, do:

```shell script
cd frontend
npm install
npm run serve
```
Then go to [http://localhost:8080/](http://localhost:8080/)

<h3>Help</h3>
I recorded a couple of short <a href="VIDEOS.md">videos</a> explaining basic steps. Also, I write some random thoughts and instructions on PrimeClue's 
<a href="http://blog.primeclue.com/">blog</a>


<h3>Contributing</h3>
I'm neither Rust nor AI expert so probably a lot of things can be improved. 
Checkout <a href="CONTRIBUTING.md">contribution rules</a> for more info.
I will gladly accept features, fixes and even one-liners to make the code more idiomatic. If you're looking for some ideas, take a look at
<a href="NEXT_FEATURES.md">next features</a> to see what's in plans.

<h3>Contact author</h3>
You can contact me at lukasz.wojtow@gmail.com
<h3>License</h3>
Affero GPL 3.0