<?php

function greet($name) {
    return "Hello " . $name;
}

class Animal {
    public $name;

    public function __construct($name) {
        $this->name = $name;
    }
}

interface Loggable {
    public function log($message);
}

