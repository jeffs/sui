const path = require('path');
console.log("START");
let action = (runtime) => {
    console.log("ACTION");
    let res = '';
    res += runtime.toString();
    // step into a function
    runtime.step(false);
    res += runtime.toString();
    // step out of a function
    runtime.stepOut(false);
    res += runtime.toString();
    // step into split coins
    runtime.step(false);
    res += runtime.toString();
    // step out of split coins
    runtime.stepOut(false);
    // step over a function
    runtime.step(true);
    res += runtime.toString();
    // step into transfer
    runtime.step(false);
    res += runtime.toString();
    return res;
};
run_spec_replay(__dirname, action);
