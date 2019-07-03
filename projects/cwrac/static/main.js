"use strict";
//1. 工具栏、当前选中的是什么工具? 基础工具：移动、点、直线、圆！ 解锁工具
//2. 不同的工具对鼠标事件响应不一样？鼠标的点击、释放

//1. 有一个数组，用于记录每个步骤干嘛
//2. 事件响应，鼠标的点击、释放
; (function () {
    function main() {
        //工具箱，枚举
        var Tools = {
            HAND: 0, //0-移动
            POINT: 1, //
            LINE: 2, //
            CIRCLE: 3
        };
        if (Object.freeze) {
            Object.freeze(Tools);
        }

        var current_tool = Tools.HAND;
        console.log(current_tool);

        var operation_list = []; //绘图记录

        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        const canvas_rect = canvas.getBoundingClientRect();

        var x = 0;
        var y = 0;
        var x1 = 0;
        var y1 = 0;

        function render() {
            if (current_tool == Tools.HAND) {
                return;
            }

            ctx.clearRect(0, 0, 800, 800); //TODO

            for (var item of operation_list) {

                if (item.type == Tools.POINT) {
                    ctx.beginPath();
                    ctx.arc(item.x, item.y, 2, 0, Math.PI * 2, true);
                    ctx.closePath();
                    ctx.stroke();
                }

                if (item.type == Tools.LINE) {
                    ctx.moveTo(item.x, item.y);
                    ctx.beginPath();
                    // ctx.strokeStyle = 'black';
                    // ctx.lineWidth = 1;
                    ctx.arc(item.x, item.y, 2, 0, Math.PI * 2, true);
                    ctx.moveTo(item.x, item.y);
                    ctx.lineTo(item.x1, item.y1);
                    ctx.arc(item.x1, item.y1, 2, 0, Math.PI * 2, true);
                    ctx.stroke();
                    ctx.closePath();
                }

                if (item.type == Tools.CIRCLE) {
                    //绘制圆心
                    ctx.beginPath();
                    ctx.arc(item.x, item.y, 2, 0, Math.PI * 2);
                    ctx.closePath();
                    ctx.stroke();

                    //绘制圆周
                    var radius = Math.sqrt((item.x - item.x1) * (item.x - item.x1) + (item.y - item.y1) * (item.y - item.y1));
                    ctx.beginPath();
                    ctx.arc(item.x, item.y, radius, 0, Math.PI * 2);
                    ctx.closePath();
                    ctx.stroke();

                }
            }

        }

        var requestAnimationFrame_status = 0;

        //事件绑定
        //浏览器大小调整
        //两种习惯： 1.鼠标点击2下确定一条直线，一个圆等 2.拖拽式？
        // https://developer.mozilla.org/en-US/docs/Web/API/Element/mousemove_event
        var isDrawing = false;
        // var x = 0;
        // var y = 0;

        // var click_x = 0;
        // var click_y = 0; 

        canvas.addEventListener('click', function (e) {
            // if (current_tool == Tools.POINT) {
            //     var x = e.clientX - canvas_rect.left;
            //     var y = e.clientY - canvas_rect.top;

            //     var point = { "type": Tools.POINT, 'x': x, 'y': y }
            //     operation_list.push(point)

            //     render()
            // }
        });
        canvas.addEventListener('mousedown', function (e) {
            var x = e.clientX - canvas_rect.left;
            var y = e.clientY - canvas_rect.top;
            isDrawing = true;

            if (current_tool != Tools.HAND) {
                var point = { "type": current_tool, 'x': x, 'y': y, 'x1': x, 'y': y };
                operation_list.push(point);
            }

            requestAnimationFrame_status = window.requestAnimationFrame(render);
        });
        canvas.addEventListener('mousemove', function (e) {
            var last_index = operation_list.length - 1;
            if (last_index < 0) {
                return;
            }

            if (isDrawing === true) {
                // draw(x, y, e.clientX - canvas_rect.left, e.clientY - canvas_rect.top);
                // x = e.clientX - canvas_rect.left;
                // y = e.clientY - canvas_rect.top;
                var x = e.clientX - canvas_rect.left;
                var y = e.clientY - canvas_rect.top;

                // var item = operation_list.pop();
                // item.x1 = x;
                // item.y1 = y;
                // operation_list.push(item);

                operation_list[last_index].x1 = x;
                operation_list[last_index].y1 = y;

                requestAnimationFrame_status = window.requestAnimationFrame(render);
            }
        });
        canvas.addEventListener('mouseup', function (e) {
            var last_index = operation_list.length - 1;
            if (last_index < 0) {
                return;
            }

            if (isDrawing === true) {
                isDrawing = false;

                var x = e.clientX - canvas_rect.left;
                var y = e.clientY - canvas_rect.top;

                // var item = operation_list.pop();
                // item.x1 = x;
                // item.y1 = y;
                // operation_list.push(item);





                // console.log(x,y);
                operation_list[last_index].x1 = x;
                operation_list[last_index].y1 = y;

                console.log(operation_list);
                console.log(last_index);



                window.cancelAnimationFrame(requestAnimationFrame_status);
            }
        });

        var toolbox = document.getElementById("toolbox");
        toolbox.addEventListener('click', function () {
            var e = event || window.event;
            if (e.target && e.target.nodeName.toUpperCase() == "INPUT") {
                current_tool = e.target.value;
            }
        }, false);

        //contextmenu?

        window.addEventListener('resize', function (e) {
            console.log(document.documentElement.clientHeight, document.documentElement.clientWidth);
        });

        //
        //
    }

    main();
})();