"use strict";
//1. 工具栏、当前选中的是什么工具? 基础工具：移动、点、直线、圆！ 解锁工具
//2. 不同的工具对鼠标事件响应不一样？鼠标的点击、释放

//1. 有一个数组，用于记录每个步骤干嘛
//2. 事件响应，鼠标的点击、释放
; (function () {
    function main() {
        //工具箱，枚举值
        var Tools = {
            HAND: 0, //0-移动
            POINT: 1, //
            LINE: 2, //
            CIRCLE: 3
        };
        if (Object.freeze) {
            Object.freeze(Tools);
        }

        var current_tool = Tools.HAND; //默认 

        var operation_list = []; //用户的绘图记录
        var intersect_points = []; //各种线条相交点

        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var canvas_rect = canvas.getBoundingClientRect();

        function get_w_b(item){
            var w = (item.y - item.y1) / (item.x - item.x1); //直线的斜率
            var b = item.y - w * item.x;  //直线的偏置项
            return [w,b];
        }

        function get_2line_intersect_point(item,item1){
            var wb = get_w_b(item);
            var wb1 = get_w_b(item1);

            // x = (b1-b0)/(a0-a1), 将x代入任一直线方程求解
            // y= a0*((b1-b0)/(a0-a1)) + b0

            var x = (wb1[1]-wb[1])/(wb[0]-wb1[0]);
            var y = wb[0] * x + wb[1];
            return {'x':x,'y':y}; 
        }

        function get_intersect_points() {
            intersect_points = [];

            //计算两个图形相交的点
            var len = operation_list.length;
            if (len < 2) {
                return;
            }
            for (var i = 0; i < len; i++) {
                var obj0 = operation_list[i];
                var obj0_wb = get_w_b(obj0);

                for (var j = i + 1; j < len; j++) {
                    if (i == j) {
                        continue;
                    }
                    if(operation_list[i].type == Tools.LINE && operation_list[j].type == Tools.LINE){
                        var point = get_2line_intersect_point(operation_list[i],operation_list[j]);
                        intersect_points.push(point); 
                    }

                    console.log(operation_list[i], operation_list[j]);
                }

            }
        }


        function calc_distance(x, y, x1, y1) {
            return Math.round(Math.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1)));
        }

        function render() {
            if (current_tool == Tools.HAND) {
                return;
            }

            get_intersect_points();

            canvas_rect = canvas.getBoundingClientRect();
            ctx.clearRect(0, 0, canvas_rect.width, canvas_rect.height); //TODO

            for (var item of intersect_points) {
                ctx.beginPath();
                ctx.arc(item.x, item.y, 2, 0, Math.PI * 2, true);
                ctx.closePath();
                ctx.stroke();
            }

            // js for 遍历数组
            //https://juejin.im/post/5a3a59e7518825698e72376b
            for (var item of operation_list) {
                //点
                if (item.type == Tools.POINT) {
                    ctx.beginPath();
                    ctx.arc(item.x, item.y, 2, 0, Math.PI * 2, true);
                    ctx.closePath();
                    ctx.stroke();
                }

                //线
                if (item.type == Tools.LINE) {
                    //需要根据用户选定的2点，确定直线斜率，画出直线

                    var w = (item.y - item.y1) / (item.x - item.x1); //直线的斜率
                    var b = item.y - w * item.x;  //直线的偏置项

                    var start = [0, Math.round(b)];
                    var end = [canvas_rect.width, Math.round(w * canvas_rect.width + b)];

                    ctx.beginPath();
                    ctx.lineWidth = 0.3;
                    ctx.strokeStyle = 'gray';
                    ctx.moveTo(start[0], start[1]);
                    ctx.lineTo(end[0], end[1]);
                    ctx.stroke();
                    ctx.closePath();

                    //绘制两点之间的线段部分
                    ctx.beginPath();
                    ctx.moveTo(item.x, item.y);
                    ctx.strokeStyle = 'black';
                    ctx.lineWidth = 1;
                    ctx.arc(item.x, item.y, 2, 0, Math.PI * 2, true);
                    ctx.moveTo(item.x, item.y);
                    ctx.lineTo(item.x1, item.y1);
                    ctx.arc(item.x1, item.y1, 2, 0, Math.PI * 2, true);
                    ctx.stroke();
                    ctx.closePath();

                }

                //圆
                if (item.type == Tools.CIRCLE) {
                    //绘制圆心
                    ctx.beginPath();
                    ctx.arc(item.x, item.y, 2, 0, Math.PI * 2);
                    ctx.closePath();
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.arc(item.x1, item.y1, 2, 0, Math.PI * 2);
                    ctx.closePath();
                    ctx.stroke();

                    //绘制圆周
                    var radius = calc_distance(item.x, item.y, item.x1, item.y1);
                    // var radius = Math.sqrt((item.x - item.x1) * (item.x - item.x1) + (item.y - item.y1) * (item.y - item.y1));
                    ctx.beginPath();
                    ctx.arc(item.x, item.y, radius, 0, Math.PI * 2);
                    ctx.closePath();
                    ctx.stroke();

                }
            }

        }

        //事件绑定
        // https://developer.mozilla.org/en-US/docs/Web/API/Element/mousemove_event 

        //绘图工具框
        var toolbox = document.getElementById("toolbox");
        toolbox.addEventListener('click', function () {
            var e = event || window.event;
            if (e.target && e.target.nodeName.toUpperCase() == "INPUT") {
                current_tool = e.target.value;
            }
        }, false);

        //两种习惯：暂时先支持第一种
        //1. 拖拽式 
        //2. 点击式 1.鼠标点击2下确定一条直线，一个圆等  
        var isDrawing = false;
        var requestAnimationFrame_status = 0;
        canvas.addEventListener('mousedown', function (e) {
            var x = e.clientX - canvas_rect.left;
            var y = e.clientY - canvas_rect.top;
            isDrawing = true;

            if (current_tool != Tools.HAND) {
                var obj = { "type": current_tool, 'x': x, 'y': y, 'x1': x, 'y': y };
                operation_list.push(obj);
            }

            requestAnimationFrame_status = window.requestAnimationFrame(render);
        });
        canvas.addEventListener('mousemove', function (e) {
            var last_index = operation_list.length - 1;
            if (last_index < 0) {
                return;
            }

            if (isDrawing === true) {
                var x = e.clientX - canvas_rect.left;
                var y = e.clientY - canvas_rect.top;

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
                operation_list[last_index].x1 = x;
                operation_list[last_index].y1 = y;

                window.cancelAnimationFrame(requestAnimationFrame_status);
            }
        });

        canvas.addEventListener('click', function (e) {
            //
        });
        //右键单击、取消？

        //浏览器大小调整
        window.addEventListener('resize', function (e) {
            var height = document.documentElement.clientHeight;
            var width = document.documentElement.clientWidth;
            canvas.height = height;
            canvas.width = width;
            requestAnimationFrame_status = window.requestAnimationFrame(render);
            // console.log(height, width);
        });

        window.addEventListener('load', function (e) {
            //页面加载，load用户的历史操作记录，呈现给用户？
        });

    }

    main();
})();